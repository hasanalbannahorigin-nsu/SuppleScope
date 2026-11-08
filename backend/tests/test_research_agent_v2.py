"""Research agent v2 and grounded extractor, with a scripted fake LLM, search and fetcher (no network)."""
import json

import pytest

from app.schemas.research import ResearchDossierV2
from app.schemas.research.common import InformationClass, ResearchStatus
from app.schemas.research.product import IdentityStatus
from app.services.extraction.grounded import GroundedExtractor
from app.services.ocr.document import ocr_document_from_text
from app.services.research_agent import prompts as P
from app.services.research_agent.budget import Budget
from app.services.research_agent.runner import run_research
from app.services.research_agent.tools.fetch import FetchedDoc
from app.services.research_agent.tools.search import SearchHit

AD_TEXT = """প্রাকৃতিক সমাধানে
ডায়বিডেক্স ক্যাপসুল
সুগার লেভেল নিয়ন্ত্রণে বিশেষ ভূমিকা পালন করে।
ইনসুলিন গ্রহণের প্রয়োজন হবে না।
অর্ডার করতে কল করুন 01758072775"""

OFFICIAL_URL = "https://examplepharma.com.bd/products/dibedex"
RETAIL_URL = "https://shop.example.net/dibedex-plus"
OFFICIAL_TEXT = """Dibedex Capsule
Manufacturer: Example Pharma Ltd., Dhaka, Bangladesh
Composition: Each capsule contains Syzygium cumini seed extract 250 mg
Gymnema sylvestre leaf extract 150 mg
Dosage: 2 capsules twice daily after meals
Warning: Not for use in pregnancy"""
RETAIL_TEXT = """Dibedex Plus Tablet by OtherBrand India
Contains Bitter melon 500 mg"""


class FakeLlm:
    model = "fake-llm"

    def __init__(self, invent_passage=False):
        self.calls = []
        self.invent_passage = invent_passage

    def parse(self, system, user, schema, *, counter=None):
        if counter is not None:
            counter.charge_llm()
        self.calls.append(schema.__name__)
        name = schema.__name__
        if name == "_Cleaned":
            text = user.split("<ocr_text>\n", 1)[1].rsplit("\n</ocr_text>", 1)[0]
            return schema(text=text, edits=[])
        if name == "_Extraction":
            return schema.model_validate({
                "product_name": {"value": "ডায়বিডেক্স", "evidence": "ডায়বিডেক্স ক্যাপসুল"},
                "ingredients": [{"name": "জিনসেং", "evidence": "জিনসেং ২০০ মি.গ্রা."}],  # not in the ad -> must be dropped
                "claims": [
                    {"text": "সুগার লেভেল নিয়ন্ত্রণে বিশেষ ভূমিকা পালন করে", "text_en": "helps control sugar level",
                     "claim_type": "TREATMENT", "evidence": "সুগার লেভেল নিয়ন্ত্রণে বিশেষ ভূমিকা পালন করে"},
                    {"text": "ইনসুলিন গ্রহণের প্রয়োজন হবে না", "text_en": "no need to take insulin",
                     "claim_type": "REPLACEMENT", "evidence": "ইনসুলিন গ্রহণের প্রয়োজন হবে না"},
                ],
                "other_text": [{"value": "01758072775", "evidence": "অর্ডার করতে কল করুন 01758072775"}],
            })
        if name == "QueryPlan":
            return schema(name_variants=["ডায়বিডেক্স", "Dibedex"], queries=["Dibedex capsule Bangladesh", "ডায়বিডেক্স"])
        if name == "Triage":
            return schema(open=[P.TriagePick(hit_index=0, expected_kind="OFFICIAL_MANUFACTURER", reason="official"),
                                P.TriagePick(hit_index=1, expected_kind="THIRD_PARTY_RETAIL", reason="retail")])
        if name == "PageReading":
            if OFFICIAL_URL in user:
                passage = "Each capsule contains Syzygium cumini seed extract 250 mg"
                return schema.model_validate({
                    "page_kind": "OFFICIAL_MANUFACTURER",
                    "about_product": {"name": "Dibedex Capsule", "manufacturer": "Example Pharma Ltd.", "country": "Bangladesh", "form": "capsule"},
                    "refers_to_ad_product": "yes", "refers_reason": "same name, Bangladesh, capsule",
                    "facts": [
                        {"field": "manufacturer_name", "value": "Example Pharma Ltd.", "passage": "Manufacturer: Example Pharma Ltd., Dhaka, Bangladesh"},
                        {"field": "dosage", "value": "2 capsules twice daily after meals", "passage": "Dosage: 2 capsules twice daily after meals"},
                        {"field": "warning", "value": "Not for use in pregnancy", "passage": "Warning: Not for use in pregnancy"},
                        # invented: the page never says this
                        {"field": "registration_number", "value": "DGDA-123", "passage": "DGDA registration no. DGDA-123"},
                    ],
                    "ingredients": [
                        {"name": "Syzygium cumini seed extract", "amount": "250", "unit": "mg", "per": "per capsule", "passage": passage},
                        {"name": "Gymnema sylvestre leaf extract", "amount": "150", "unit": "mg", "passage": "Gymnema sylvestre leaf extract 150 mg"},
                        # wrong amount: passage says 150 mg
                        {"name": "Gymnema sylvestre leaf extract", "amount": "500", "unit": "mg", "passage": "Gymnema sylvestre leaf extract 150 mg"},
                    ],
                })
            return schema.model_validate({
                "page_kind": "THIRD_PARTY_RETAIL",
                "about_product": {"name": "Dibedex Plus Tablet", "brand": "OtherBrand", "country": "India", "form": "tablet"},
                "refers_to_ad_product": "no", "refers_reason": "different brand, country and form",
                "ingredients": [{"name": "Bitter melon", "amount": "500", "unit": "mg", "passage": "Contains Bitter melon 500 mg"}],
            })
        if name == "CheckResult":
            items = json.loads(user)
            return schema(items=[P.CheckItem(index=i["index"], supports=True) for i in items])
        if name == "FollowUp":
            return schema(queries=[])
        raise AssertionError(f"unexpected schema {name}")


class FakeSearch:
    name = "fake"

    def __init__(self):
        self.queries = []

    def search(self, query, max_results=8):
        self.queries.append(query)
        return [SearchHit("Dibedex Capsule - Example Pharma", OFFICIAL_URL, "Ayurvedic capsule", "fake", 1),
                SearchHit("Dibedex Plus Tablet", RETAIL_URL, "OtherBrand", "fake", 2)]


def fake_fetch(url, max_chars=60000):
    text = OFFICIAL_TEXT if url == OFFICIAL_URL else RETAIL_TEXT
    return FetchedDoc(url=url, final_url=url, title="", text=text, content_type="text/html", sha256="x", status=200)


def _budget(**kw):
    args = dict(max_search_calls=10, max_fetches=10, max_llm_calls=30, max_retries_per_call=0, max_seconds=60, max_rounds=2)
    args.update(kw)
    return Budget(**args)


def _ad():
    doc = ocr_document_from_text("ad_test", AD_TEXT)
    return GroundedExtractor(FakeLlm()).run(doc).extraction


def test_grounded_extractor_keeps_ad_text_only():
    ad = _ad()
    assert ad.product_name.value == "ডায়বিডেক্স"
    assert ad.product_name.span.line_ids == ["seg01_l002"]
    assert [c.claim_id for c in ad.claims] == ["c01", "c02"]
    assert ad.ingredients == []  # the invented ingredient was dropped ...
    assert ad.rejected and ad.rejected[0].path == "ingredients[0]"  # ... and reported


def test_agent_identifies_product_and_official_ingredients():
    llm = FakeLlm()
    d = run_research(_ad(), llm=llm, search=FakeSearch(), fetcher=fake_fetch, budget=_budget(), use_cache=False)
    ResearchDossierV2.model_validate_json(d.model_dump_json())  # round-trips and passes reference checks
    ident = d.product.identity
    assert ident.status == IdentityStatus.VERIFIED
    chosen = next(c for c in ident.candidates if c.candidate_id == ident.chosen_candidate_id)
    assert chosen.name == "Dibedex Capsule"
    lookalike = next(c for c in ident.candidates if c.name == "Dibedex Plus Tablet")
    assert lookalike.rejected_reason
    comp = d.product.composition
    assert comp.status == ResearchStatus.FOUND
    assert [i.name for i in comp.ingredients] == ["Syzygium cumini seed extract", "Gymnema sylvestre leaf extract"]
    assert comp.ingredients[0].amount == "250" and comp.ingredients[0].per == "per capsule"
    assert d.product.brand_manufacturer.official_website == OFFICIAL_URL
    # look-alike's ingredient is not in the composition and is marked as not about this product
    bitter = next(f for f in d.findings if f.value == "Bitter melon")
    assert bitter.validation.refers_to_product is False and bitter.information_class == InformationClass.UNVERIFIED
    # invented passage and wrong amount are rejected
    reg = next(f for f in d.findings if f.field == "registration_number")
    assert not reg.validation.quote_in_source and reg.information_class == InformationClass.UNVERIFIED
    wrong = next(f for f in d.findings if f.field == "ingredient" and f.unit == "500 mg" and "Gymnema" in f.value)
    assert wrong.validation.value_in_passage is False
    assert d.product.regulatory == []
    # the ad names no ingredients: discrepancy recorded
    assert any(x.kind.value == "INGREDIENT_IN_OFFICIAL_NOT_IN_AD" for x in d.product.ad_vs_official)
    # later phases are honest about not being done
    assert {m.item for m in d.missing_information if m.status == ResearchStatus.NOT_SEARCHED} >= {"claim-by-claim evidence"}
    assert d.budget.used_search_calls <= 10 and not d.budget.exhausted
    assert all(e.phase for e in d.research_log)


def test_budget_exhaustion_still_returns_valid_dossier():
    d = run_research(_ad(), llm=FakeLlm(), search=FakeSearch(), fetcher=fake_fetch,
                     budget=_budget(max_search_calls=0), use_cache=False)
    assert d.budget.exhausted and d.budget.exhausted_reason == "search"
    assert d.product.identity.status == IdentityStatus.NOT_FOUND
    assert any("stopped early" in u for u in d.uncertainties)
    assert any(m.status == ResearchStatus.NOT_SEARCHED for m in d.missing_information)


def test_no_product_name_means_no_search():
    from app.schemas.research.ad_extraction import AdExtraction, ExtractorInfo
    ad = AdExtraction(ad_id="x", extractor=ExtractorInfo(name="t", version="0"))
    search = FakeSearch()
    d = run_research(ad, llm=FakeLlm(), search=search, fetcher=fake_fetch, budget=_budget(), use_cache=False)
    assert search.queries == []
    assert d.product.identity.status == IdentityStatus.NOT_FOUND


def test_requires_llm(monkeypatch):
    monkeypatch.setattr("app.services.research_agent.runner.llm_from_settings", lambda: None)
    with pytest.raises(RuntimeError, match="needs an LLM"):
        run_research(_ad(), search=FakeSearch(), fetcher=fake_fetch, budget=_budget(), use_cache=False)


def test_search_provider_is_configurable(monkeypatch):
    from app.services.research_agent.tools import search as S
    monkeypatch.setattr(S, "get_settings", lambda: type("X", (), {"web_search_provider": "tavily", "tavily_api_key": "k"})())
    assert S.provider_from_settings().name == "tavily"
    monkeypatch.setattr(S, "get_settings", lambda: type("X", (), {"web_search_provider": "duckduckgo", "tavily_api_key": ""})())
    assert S.provider_from_settings().name == "duckduckgo"
