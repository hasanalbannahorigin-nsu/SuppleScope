"""Completed stack, research side: literature filter, registry, openFDA, trafilatura, merged search, product
sources, dictionary normalisation, iDISK2.0, fact sheets and the reference phase. All offline (fixtures)."""
import json

import pytest

from app.schemas.research import ResearchDossierV2
from app.schemas.research.common import InformationClass, ResearchStatus, SourceType, StudyType
from app.services.knowledge import factsheets as F
from app.services.knowledge.idisk import IdiskStore, build_database, discover, relation_kind
from app.services.knowledge.normalise import IngredientNormaliser, split_form
from app.services.research_agent.budget import Budget
from app.services.research_agent.reference_phase import ReferencePhase
from app.services.research_agent.runner import run_research
from app.services.research_agent.state import ResearchState
from app.services.research_agent.tools import openfda
from app.services.research_agent.tools import product_sources as PS
from app.services.research_agent.tools.fetch import page_text
from app.services.research_agent.tools.literature import (
    PUBMED_STUDY_FILTER,
    literature_providers_from_settings,
    parse_clinicaltrials,
    with_study_filter,
)
from app.services.research_agent.tools.search import MergedSearch, SearchHit

from tests.test_research_agent_v2 import FakeLlm, FakeSearch, _ad, _budget, fake_fetch, fake_pubmed

CT_PAYLOAD = {"studies": [{
    "protocolSection": {
        "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Jamun Seed Powder in Prediabetes"},
        "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2019-03"}},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "enrollmentInfo": {"count": 60},
                         "designInfo": {"allocation": "RANDOMIZED"}},
        "conditionsModule": {"conditions": ["Prediabetes"]},
        "armsInterventionsModule": {"interventions": [{"type": "DIETARY_SUPPLEMENT", "name": "Syzygium cumini seed powder"}]},
        "descriptionModule": {"briefSummary": "This trial tests whether jamun seed powder lowers fasting glucose."},
    },
    "hasResults": False,
}]}


# ---- literature -------------------------------------------------------------------------------------------

def test_pubmed_study_filter_and_registry_parsing():
    assert with_study_filter("pubmed", '("x"[tiab])').endswith(PUBMED_STUDY_FILTER)
    assert with_study_filter("europepmc", "x") is None
    papers = parse_clinicaltrials(CT_PAYLOAD)
    assert len(papers) == 1
    p = papers[0]
    assert p.nct_id == "NCT01234567" and p.provider == "clinicaltrials" and p.has_results is False
    assert p.url == "https://clinicaltrials.gov/study/NCT01234567" and p.year == 2019
    assert "Enrollment: 60." in p.abstract and "allocation randomized" in p.abstract
    assert "Results posted on the registry: no." in p.abstract
    assert "clinicaltrials" not in literature_providers_from_settings()  # never a fallback for papers


def test_filtered_search_falls_back_to_unfiltered():
    from app.services.research_agent.evidence_phase import EvidencePhases
    seen = []

    def pubmed(query, retmax=5):
        seen.append(query)
        return [] if PUBMED_STUDY_FILTER in query else fake_pubmed(query, retmax)

    state = ResearchState(budget=_budget(), llm=FakeLlm(), search_provider=FakeSearch())
    state.literature_search = {"pubmed": pubmed}
    ph = EvidencePhases.__new__(EvidencePhases)
    ph.s, ph.providers, ph.papers_per_query, ph.study_filter = state, ["pubmed"], 5, True
    papers, tried = ph._search(["Syzygium cumini"], ["blood glucose"], "ingredient", filtered=True)
    assert papers and len(seen) == 2 and PUBMED_STUDY_FILTER in seen[0] and PUBMED_STUDY_FILTER not in seen[1]
    assert len(tried) == 2


# ---- openFDA ------------------------------------------------------------------------------------------------

def test_openfda_caers_and_label_parsing():
    rep = openfda.parse_food_events({"meta": {"results": {"total": 3}}, "results": [
        {"reactions": ["NAUSEA", "Headache"], "outcomes": ["Other Outcome"], "products": [{"name_brand": "CENTRUM SILVER"}]},
        {"reactions": ["nausea"], "outcomes": ["Hospitalization"], "products": [{"name_brand": "CENTRUM"}]},
    ]}, 'products.name_brand:"Centrum"')
    assert rep.total_reports == 3 and rep.reports_read == 2 and rep.top_reactions[0] == ("nausea", 2)
    text = openfda.caers_text(rep)
    assert "not proof" in text and "Reaction reported: nausea (2 report(s))." in text
    mentions = openfda.parse_label_mentions({"results": [{
        "set_id": "abc", "openfda": {"generic_name": ["WARFARIN SODIUM"]},
        "drug_interactions": ["Botanicals: Some botanicals may increase the effects. Garlic and ginkgo biloba may "
                              "increase bleeding risk. Other text here."],
    }]}, ["Garlic"])
    assert len(mentions) == 1
    m = mentions[0]
    assert m.drug == "warfarin sodium" and m.section == "drug_interactions" and m.sentence.startswith("Garlic and ginkgo")
    assert "setid=abc" in m.url
    assert openfda.mentions_from_json(openfda.mentions_to_json(mentions)) == mentions
    assert openfda.reports_from_json(openfda.reports_to_json(rep)) == rep


def test_label_mentions_match_abbreviated_and_curly_names():
    # Real labels print "St. John\u2019s Wort" (curly apostrophe), and "St." must not end the sentence.
    mentions = openfda.parse_label_mentions({"results": [{
        "set_id": "x1", "openfda": {"generic_name": ["CYCLOSPORINE"]},
        "drug_interactions": ["7.1 Inducers. Drugs such as rifampin and St. John\u2019s Wort may decrease "
                              "cyclosporine concentrations. Monitor levels."],
        "warnings": ["Avoid st johns-wort products during treatment. Other text."],
    }]}, ["St. John's Wort"])
    by_section = {m.section: m.sentence for m in mentions}
    assert by_section["drug_interactions"] == ("Drugs such as rifampin and St. John\u2019s Wort may decrease "
                                               "cyclosporine concentrations.")
    assert by_section["warnings"] == "Avoid st johns-wort products during treatment."
    assert openfda.label_query(["St. John's Wort", "Garlic"], ("drug_interactions", "warnings")) == (
        'drug_interactions:"St. John\'s Wort" OR drug_interactions:"Garlic" OR warnings:"St. John\'s Wort" OR '
        'warnings:"Garlic"')


def test_label_without_openfda_block_and_table_text():
    table = ("Drugs/Dietary Supplements That Decrease Cyclosporine Concentrations Antibiotics nafcillin rifampin "
             * 12 + "Herbal Preparations St. John's Wort " + "Anticonvulsants carbamazepine oxcarbazepine " * 12)
    m = openfda.parse_label_mentions({"results": [{
        "set_id": "t1", "openfda": {},
        "spl_product_data_elements": ["Cyclosporine Cyclosporine CYCLOSPORINE ALCOHOL CORN OIL"],
        "drug_interactions": [table]}]}, ["St. John's Wort"])[0]
    assert m.drug == "cyclosporine"
    assert "St. John's Wort" in m.sentence and len(m.sentence) <= 600 and m.sentence in table
    assert not m.sentence.startswith(" ") and m.sentence.split()[0] in table.split()
    assert openfda.label_drug_name({"spl_product_data_elements": ["Sandimmune cyclosporine CYCLOSPORINE"]}) == \
        "sandimmune cyclosporine"
    assert openfda.label_drug_name({}) == "unnamed drug"


def test_label_search_sends_ascii_only_and_retries_simple_query_on_400(monkeypatch):
    import httpx

    sent = []

    def fake_get(path, params, attempts=3):
        sent.append(params["search"])
        if len(sent) == 1:
            req = httpx.Request("GET", "https://api.fda.gov/drug/label.json")
            raise httpx.HTTPStatusError("400", request=req, response=httpx.Response(400, request=req))
        return {"results": [{"set_id": "s", "openfda": {"generic_name": ["WARFARIN"]},
                             "drug_interactions": ["Garlic may increase bleeding."]}]}

    monkeypatch.setattr(openfda, "_get", fake_get)
    out = openfda.drug_label_mentions(["Garlic", "রসুন", "Ginkgo"])
    assert "রসুন" not in sent[0] and sent[0].count(" OR ") == 5
    assert sent[1] == 'drug_interactions:"Garlic"'
    assert [m.sentence for m in out] == ["Garlic may increase bleeding."]


# ---- web ----------------------------------------------------------------------------------------------------

def test_page_text_keeps_main_content_and_side_tables():
    html = ("<html><head><title>Dibedex</title></head><body><nav>Home | Shop</nav><article><h1>Dibedex Capsule</h1>"
            "<p>Dibedex is an Ayurvedic capsule made in Bangladesh by Example Pharma for people who want to manage "
            "blood sugar alongside their usual care.</p><p>Take two capsules twice daily after meals with water. "
            "Keep out of reach of children and store below 30 degrees.</p></article>"
            "<aside><table><tr><td>Composition</td><td>Syzygium cumini 250 mg</td></tr></table></aside></body></html>")
    title, text = page_text(html)
    assert title == "Dibedex"
    assert "Take two capsules twice daily" in text and "Syzygium cumini 250 mg" in text
    assert text.count("Take two capsules twice daily") == 1


class _P:
    def __init__(self, name, hits=None, fail=False):
        self.name, self.hits, self.fail = name, hits or [], fail

    def search(self, query, max_results=8):
        if self.fail:
            raise RuntimeError("down")
        return self.hits


def test_merged_search_interleaves_and_dedupes():
    a = _P("duckduckgo", [SearchHit("A1", "https://www.x.com/a/", None, "duckduckgo", 1),
                          SearchHit("A2", "https://y.com/b", None, "duckduckgo", 2)])
    b = _P("tavily", [SearchHit("B1", "https://x.com/a", None, "tavily", 1),
                      SearchHit("B2", "https://z.com/c", None, "tavily", 2)])
    m = MergedSearch([a, b])
    hits = m.search("q", 8)
    assert [h.url for h in hits] == ["https://www.x.com/a/", "https://y.com/b", "https://z.com/c"]
    assert [h.rank for h in hits] == [1, 2, 3] and m.name == "merged(duckduckgo+tavily)"
    assert MergedSearch([_P("a", fail=True), b]).search("q")[0].title == "B1"  # one provider down
    with pytest.raises(RuntimeError):
        MergedSearch([_P("a", fail=True)]).search("q")


# ---- product sources ----------------------------------------------------------------------------------------

DSLD_LABEL = {
    "id": 12345, "fullName": "Centrum Silver Adults 50+", "brandName": "Centrum",
    "contacts": [{"types": ["Distributor"], "contactDetails": {"name": "Pfizer Inc.", "city": "New York", "country": "USA"}}],
    "servingSizes": [{"minQuantity": 1, "unit": "Tablet(s)"}],
    "ingredientRows": [
        {"name": "Vitamin D3", "quantity": [{"quantity": 25, "unit": "mcg"}], "notes": "as cholecalciferol"},
        {"name": "Zinc", "quantity": [{"quantity": 11, "unit": "mg"}]},
    ],
    "otheringredients": {"text": "Calcium Carbonate, Microcrystalline Cellulose"},
    "statements": [{"type": "Precautions", "notes": "Keep out of reach of children."}],
}


def test_dsld_hits_and_label_text(monkeypatch):
    hits = PS.parse_dsld_hits({"hits": [{"_id": "12345", "_source": {"fullName": "Centrum Silver", "brandName": "Centrum"}},
                                        {"_id": "x"}]})
    assert [h.url for h in hits] == ["https://dsld.od.nih.gov/label/12345"]
    assert hits[0].title == "DSLD label: Centrum Silver" and "brand Centrum" in hits[0].snippet
    title, text = PS.dsld_label_text(DSLD_LABEL)
    assert "- Vitamin D3: 25 mcg (as cholecalciferol)" in text and "- Zinc: 11 mg" in text
    assert "Distributor: Pfizer Inc., New York, USA." in text and "Other ingredients: Calcium Carbonate" in text
    assert "Serving size: 1 Tablet(s)." in text
    monkeypatch.setattr(PS, "_get_json", lambda path, params: DSLD_LABEL)
    doc = PS.fetch_any("https://dsld.od.nih.gov/label/12345")
    assert doc.title.startswith("DSLD label") and "Vitamin D3" in doc.text
    called = []
    monkeypatch.setattr(PS, "fetch_document", lambda url, **kw: called.append(url) or doc)
    PS.fetch_any("https://example.com/p")
    assert called == ["https://example.com/p"]
    qs = PS.extra_queries("Dibedex", None, {"dgda", "pharmacy"})
    assert qs[0] == '"Dibedex" site:dgda.gov.bd' and "site:arogga.com" in qs[1]
    assert PS.extra_queries("Dibedex", None, set()) == []


def test_product_phase_adds_dsld_and_bd_queries():
    from app.services.research_agent.product_phase import ProductPhase

    class Dsld:
        name = "dsld"

        def __init__(self):
            self.queries = []

        def search(self, q, max_results=5):
            self.queries.append(q)
            return [SearchHit("DSLD label: Dibedex", "https://dsld.od.nih.gov/label/1", None, "dsld", 1)]

    search, dsld = FakeSearch(), Dsld()
    state = ResearchState(budget=_budget(), llm=FakeLlm(), search_provider=search)
    ph = ProductPhase(state, _ad(), product_sources={"dsld", "dgda", "pharmacy"}, dsld=dsld)
    ph.name_variants.append("Dibedex")
    hits = ph._product_source_hits()
    assert any("site:dgda.gov.bd" in q for q in search.queries) and any("site:medeasy.health" in q for q in search.queries)
    assert dsld.queries == ["Dibedex"] and any(h.provider == "dsld" for h in hits)
    assert any(q.startswith("dsld: ") for q in state.queries_tried)


# ---- name normalisation ---------------------------------------------------------------------------------------

def test_bangla_names_normalise_to_latin():
    n = IngredientNormaliser()
    r = n.normalise("জামের বীজ")
    assert r.scientific == "Syzygium cumini" and r.english == "Jamun" and r.method == "exact"
    assert n.normalise("কালোজিরা").scientific == "Nigella sativa"
    assert n.normalise("মেথি চূর্ণ").scientific == "Trigonella foenum-graecum"
    assert n.normalise("Ashwagandha root extract").scientific == "Withania somnifera"
    fz = n.normalise("অশ্বগন্ধ্যা")  # misspelt
    assert fz.method == "fuzzy" and fz.scientific == "Withania somnifera"
    assert n.normalise("ঘোড়ার ডিম").entry_id is None  # unknown -> nothing invented
    assert n.same_ingredient("জাম", "Syzygium cumini seed extract")
    assert not n.same_ingredient("জাম", "Gymnema sylvestre leaf extract")
    assert split_form("Gymnema sylvestre leaf extract") == ("Gymnema sylvestre", "leaf extract")


# ---- iDISK2.0 ---------------------------------------------------------------------------------------------------

def _idisk_folder(tmp_path):
    d = tmp_path / "idisk"
    d.mkdir()
    (d / "dsi.csv").write_text("dsi_id,name,synonyms,safety\nDSI001,Syzygium cumini,jamun|java plum,"
                               "May lower blood glucose; monitor with diabetes medicines.\nDSI002,Garlic,Allium sativum,\n",
                               encoding="utf-8")
    (d / "drug.csv").write_text("drug_id,name\nDRG01,Insulin\nDRG02,Warfarin\n", encoding="utf-8")
    (d / "disease.csv").write_text("disease_id,name\nDIS01,Type 2 diabetes\n", encoding="utf-8")
    (d / "relationships.csv").write_text("head,relation,tail\nDSI001,interacts_with,DRG01\nDSI001,is_effective_for,DIS01\n"
                                         "DSI002,interacts_with,DRG02\n", encoding="utf-8")
    return d


def test_idisk_loader_guesses_columns_and_answers_lookups(tmp_path):
    folder = _idisk_folder(tmp_path)
    maps = {m.path.split("/")[-1]: m for m in discover(folder)}
    assert maps["dsi.csv"].id_col == "dsi_id" and maps["dsi.csv"].synonym_cols == ["synonyms"]
    assert maps["dsi.csv"].text_cols == ["safety"] and maps["relationships.csv"].kind == "relations"
    counts = build_database(folder, tmp_path / "idisk.sqlite")
    assert counts == {"entities": 5, "names": 8, "relations": 3}
    store = IdiskStore(tmp_path / "idisk.sqlite")
    assert store.match(["jamun"]).entity_id == "DSI001"
    assert store.match(["Syzygium cumini seed"], threshold=80).entity_id == "DSI001"  # fuzzy
    facts = {f.kind: f for f in store.facts("DSI001")}
    assert facts["interaction"].value == "Insulin" and facts["effective_for"].value == "Type 2 diabetes"
    assert "monitor with diabetes medicines" in facts["attribute"].text
    assert relation_kind("has_adverse_effect_on") == "adverse_effect"


def test_idisk_1x_rrf_files(tmp_path):
    d = tmp_path / "rrf"
    d.mkdir()
    (d / "MRCONSO.RRF").write_text("DC0001|A1|Fenugreek|CN|NMCD|x|Y|\nDC0001|A2|Trigonella foenum-graecum|SN|NMCD|x|N|\n"
                                   "DC0009|A3|Warfarin|CN|NMCD|x|Y|\n", encoding="utf-8")
    (d / "MRSTY.RRF").write_text("DC0001|SDSI|\nDC0009|SPD|\n", encoding="utf-8")
    (d / "MRREL.RRF").write_text("R1|DC0001|interacts_with|DC0009|NMCD|\n", encoding="utf-8")
    build_database(d, tmp_path / "rrf.sqlite")
    store = IdiskStore(tmp_path / "rrf.sqlite")
    m = store.match(["Trigonella foenum-graecum"])
    assert m.entity_id == "DC0001" and m.name == "Fenugreek" and m.entity_type == "DSI"
    assert [f.value for f in store.facts("DC0001")] == ["Warfarin"]
    assert store.stats()["format"] == "iDISK 1.x RRF"


# ---- fact sheets --------------------------------------------------------------------------------------------------

def test_fact_sheet_safety_sentences_and_xml():
    text = ("Ashwagandha is an evergreen shrub. Ashwagandha might interact with thyroid hormone medicines and "
            "diabetes medicines. People who are pregnant should avoid ashwagandha because it might cause miscarriage. "
            "It grows in India.")
    s = F.safety_sentences(text)
    assert len(s) == 2 and s[0].startswith("Ashwagandha might interact")
    xml = "<Factsheet><Title>Ashwagandha</Title><Content>&lt;p&gt;" + text + "&lt;/p&gt;</Content></Factsheet>"
    title, body = F._xml_text(xml)
    assert title == "Ashwagandha" and "might interact with thyroid" in body
    assert F.ods_resource("Vitamin D") == "VitaminD" and F.ods_resource("জাম") is None
    assert F.nccih_slug("Licorice") == "licorice-root" and F.nccih_slug("Black seed") == "nigella-sativa"


def test_fact_sheet_bullets_are_separate_sentences():
    text = ("- Withanolides have been associated with anti-inflammatory and antioxidant effects.\n"
            "- Ashwagandha might interact with thyroid hormone medicines and sedatives.")
    assert F.safety_sentences(text) == ["Ashwagandha might interact with thyroid hormone medicines and sedatives."]


def test_body_parts_are_not_ingredients():
    from app.services.knowledge.normalise import looks_like_body_part, not_an_ingredient
    for name in ("pancreatic beta cell", "Pancreatic beta-cells", "blood sugar level", "insulin sensitivity", "liver"):
        assert looks_like_body_part(name), name
    for name in ("cod liver oil", "beef liver", "pancreatin", "glucose", "insulin", "Fenugreek", "জাম বিচি", ""):
        assert not looks_like_body_part(name), name
    assert not_an_ingredient("pancreatic beta cell", "অগ্ন্যাশয়ের বিটা কোষ")
    assert not not_an_ingredient("pancreas", "Bovine pancreas") and not not_an_ingredient(None, "মেথি")


def test_fact_sheet_drops_citations_and_efficacy_lines():
    assert not F.is_safety_statement("Bethesda, MD: National Institute of Diabetes and Digestive and Kidney Diseases; 2019.")
    assert not F.is_safety_statement("There is not enough high-quality evidence to determine whether fenugreek is useful "
                                     "for diabetes or menstrual cramps.")
    assert F.is_safety_statement("Fenugreek may be unsafe for use during pregnancy.")
    assert F.is_safety_statement("People with thyroid disease should talk with their health care provider first.")
    assert not F.states_concern("Fenugreek seed is used for diabetes.") and F.states_concern("Avoid it in pregnancy.")


def test_ods_falls_back_to_the_html_page_when_the_api_refuses(monkeypatch):
    import httpx

    page_html = ("<html><head><title>Iron - Health Professional Fact Sheet</title></head><body><main><p>"
                 + "Iron is a mineral that is naturally present in many foods. " * 12
                 + "Iron supplements can interact with levodopa, a medicine for Parkinson disease."
                 + "</p></main></body></html>")
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        if "api/index.aspx" in url:
            req = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("403 Forbidden", request=req, response=httpx.Response(403, request=req))
        return httpx.Response(200, text=page_html, request=httpx.Request("GET", url))

    monkeypatch.setattr(F, "_get", fake_get)
    sheet = F.ods_fact_sheet("Iron")
    assert "readinglevel=Health%20Professional" in calls[0]
    assert sheet and sheet.retrieved_from == "https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/"
    assert any("levodopa" in x for x in sheet.safety_sentences)

    def always_forbidden(url, params=None):
        req = httpx.Request("GET", url)
        raise httpx.HTTPStatusError("403 Forbidden", request=req, response=httpx.Response(403, request=req))

    monkeypatch.setattr(F, "_get", always_forbidden)
    with pytest.raises(httpx.HTTPStatusError):
        F.ods_fact_sheet("Iron")  # the research log records why ODS is missing


# ---- reference phase in a full run --------------------------------------------------------------------------------

class RegistryLlm(FakeLlm):
    """FakeLlm that also reads the registry record (its passage is a line of the registry summary)."""

    def parse(self, system, user, schema, *, counter=None):
        if schema.__name__ == "PapersReading" and "Jamun Seed Powder in Prediabetes" in user:
            if counter is not None:
                counter.charge_llm()
            data = json.loads(user)
            idx = next(p["index"] for p in data["papers"] if p["title"] == "Jamun Seed Powder in Prediabetes")
            base = super().parse(system, user, schema)
            base.papers.append(schema.model_validate({"papers": [{
                "index": idx, "relevant": True, "study_type": "RCT", "human_study": True, "effects": ["blood glucose"],
                "passage": "This trial tests whether jamun seed powder lowers fasting glucose."}]}).papers[0])
            return base
        return super().parse(system, user, schema, counter=counter)


def test_full_run_with_registry_references_and_caers(tmp_path, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "research_openfda", True)
    folder = _idisk_folder(tmp_path)
    build_database(folder, tmp_path / "idisk.sqlite")
    sheet = F.FactSheet(database="NIH ODS", title="Jamun", url="https://ods.example/jamun", text="x" * 600,
                        safety_sentences=["Jamun might lower blood sugar too much when taken with insulin."])
    label = [openfda.LabelMention(drug="glipizide", section="warnings", sentence="Syzygium cumini may add to the effect.",
                                  set_id="s1", url="https://dailymed.example/s1")]
    ref_state = {}

    def make_ref(state):
        ref_state["phase"] = ReferencePhase(state, idisk=IdiskStore(tmp_path / "idisk.sqlite"), sources={"idisk", "ods"},
                                            use_openfda=True, sheet_fetchers={"ods": lambda n: sheet if n == "Jamun" else None},
                                            label_search=lambda names: label if "Syzygium cumini" in names else [])
        return ref_state["phase"]

    monkeypatch.setattr("app.services.research_agent.runner.ReferencePhase", make_ref)
    monkeypatch.setattr(openfda, "food_event_reports", lambda name: openfda.parse_food_events(
        {"meta": {"results": {"total": 1}}, "results": [{"reactions": ["nausea"], "products": [{"name_brand": "DIBEDEX"}]}]},
        f'products.name_brand:"{name}"'))
    registry = {"pubmed": fake_pubmed, "clinicaltrials": lambda q, retmax=3: parse_clinicaltrials(CT_PAYLOAD)
                if "Syzygium" in q else []}
    monkeypatch.setattr(get_settings(), "research_trial_registry", True)
    budget = Budget(max_search_calls=10, max_fetches=10, max_llm_calls=40, max_retries_per_call=0, max_seconds=60,
                    max_rounds=2, max_literature_calls=30, max_lookup_calls=10)
    d = run_research(_ad(), llm=RegistryLlm(), search=FakeSearch(), fetcher=fake_fetch, literature=registry,
                     budget=budget, use_cache=False)
    ResearchDossierV2.model_validate_json(d.model_dump_json())
    syz = next(i for i in d.ingredients if "Syzygium" in i.name)
    # registry record: kept, typed, never more than CONTEXT for a claim
    reg = next(e for e in d.evidence if "jamun seed powder lowers fasting glucose" in e.passage)
    assert reg.study_type == StudyType.REGISTERED_TRIAL_NO_RESULTS and reg.validation.passed
    src = next(s for s in d.sources if s.source_id == reg.source_id)
    assert src.source_type == SourceType.CLINICAL_TRIAL_REGISTRY and src.nct_id == "NCT01234567"
    links = [x for c in d.claims for x in c.evidence_links if x.evidence_id == reg.evidence_id]
    assert all(x.relation.value == "CONTEXT" for x in links)
    # references: dictionary + iDISK2.0 + ODS + drug label, each with a REFERENCE finding
    dbs = {k.database for k in syz.knowledge_links}
    assert {"SuppleScope dictionary", "iDISK2.0", "NIH ODS"} <= dbs
    idisk_link = next(k for k in syz.knowledge_links if k.database == "iDISK2.0")
    assert idisk_link.entity_id == "DSI001"
    refs = [f for f in d.findings if f.finding_id in syz.reference_finding_ids]
    assert refs and all(f.information_class == InformationClass.REFERENCE and f.validation.passed for f in refs)
    drugs = {x.drug_or_class for x in syz.safety.drug_interactions}
    assert {"insulin", "glipizide"} <= drugs
    # CAERS for the finished product
    ae = d.product.finished_product.adverse_event_reports
    assert ae is not None and ae.total_reports == 1 and ae.top_reactions[0].name == "nausea" and "not proof" in ae.caveat
    assert d.budget.used_lookup_calls >= 2
    assert d.schema_version == "2.2.0"


def test_reference_phase_survives_lookup_budget(tmp_path):
    state = ResearchState(budget=Budget(max_search_calls=1, max_fetches=1, max_llm_calls=1, max_retries_per_call=0,
                                        max_seconds=60, max_lookup_calls=0), llm=FakeLlm(), search_provider=FakeSearch())
    from app.schemas.research.ingredient import IngredientResearch, IngredientSafety
    ing = IngredientResearch(ingredient_id="i1", name="জাম", origin="AD_ONLY", status=ResearchStatus.NOT_SEARCHED,
                             safety=IngredientSafety(status=ResearchStatus.NOT_SEARCHED))
    ph = ReferencePhase(state, idisk=None, sources={"ods"}, use_openfda=False,
                        sheet_fetchers={"ods": lambda n: pytest.fail("must not be called")})
    out = ph.enrich(ing, [])
    assert [k.database for k in out.knowledge_links] == ["SuppleScope dictionary"]
    assert any("Reference lookups stopped early" in u for u in state.uncertainties)
