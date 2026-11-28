"""Extractor 1.4: repeated claims are merged, English pack claims and quality/approval lines are claims.

On the 7 real ads (2026-11-28) the LLM put "১০০% ন্যাচারাল", "GMP সার্টিফাইড", "কেমিক্যাল মুক্ত", "১০০% গ্যারান্টি" and
"বিএসটিআই ... অনুমোদিত" in other_text, skipped Centrum's English pack claims, and listed "শক্তি বাড়ায়" twice.
"""
from app.schemas.research.ad_extraction import AdExtraction
from app.services.extraction.grounded import EXTRACT_SYSTEM, GroundedExtractor, _Claim, _Extraction, merge_repeated_claims
from app.services.ocr.document import ocr_document_from_text
from app.services.text.bangla import claim_key

AD = "৭ দিনে\nওজন বাড়ান!\nশক্তি বাড়ায়\n7 দিনে দৃশ্যমান পরিবর্তন!\nওজন বাড়ায়\nশক্তি ও স্ট্যামিনা বাড়ায়\nGMP সার্টিফাইড\nSupports heart, brain, eye"


def _c(text, evidence, qualifiers=(), negated=False, claim_type="IMPROVEMENT"):
    return _Claim(text=text, claim_type=claim_type, qualifiers=list(qualifiers), negated=negated, evidence=evidence)


def test_claim_key_ignores_verb_ending_but_not_the_target():
    assert claim_key("ওজন বাড়ান!") == claim_key("ওজন বাড়ায়")
    assert claim_key("হাড় মজবুত করে") != claim_key("দাঁত মজবুত করে")
    assert claim_key("ওজন বাড়ায়") != claim_key("ওজন কমায়")


def test_repeated_claim_is_kept_once_with_all_qualifiers():
    ex = _Extraction(claims=[
        _c("ওজন বাড়ান!", "৭ দিনে\nওজন বাড়ান!", ["৭ দিনে"]),
        _c("শক্তি বাড়ায়", "শক্তি বাড়ায়"),
        _c("ওজন বাড়ায়", "7 দিনে দৃশ্যমান পরিবর্তন!\nওজন বাড়ায়", ["7 দিনে", "দৃশ্যমান পরিবর্তন"]),
        _c("শক্তি বাড়ায়", "শক্তি ও স্ট্যামিনা বাড়ায়"),
        _c("স্ট্যামিনা বাড়ায়", "শক্তি ও স্ট্যামিনা বাড়ায়"),
        _c("ওজন বাড়ায়", "ওজন বাড়ায়", negated=True),  # "does not increase weight" is a different claim
    ])
    out = merge_repeated_claims(ex)
    assert [c.text for c in out.claims] == ["ওজন বাড়ান!", "শক্তি বাড়ায়", "স্ট্যামিনা বাড়ায়", "ওজন বাড়ায়"]
    assert out.claims[0].qualifiers == ["৭ দিনে", "দৃশ্যমান পরিবর্তন"]  # "7 দিনে" is the same as "৭ দিনে"
    assert out.claims[0].evidence == "৭ দিনে\nওজন বাড়ান!"
    assert ex.claims[0].qualifiers == ["৭ দিনে"]  # the input is not changed


class _Llm:
    model = "fake"

    def parse(self, system, user, schema, *, counter=None):
        return schema.model_validate({"claims": [
            {"text": "ওজন বাড়ান!", "claim_type": "IMPROVEMENT", "qualifiers": ["৭ দিনে"], "evidence": "৭ দিনে\nওজন বাড়ান!"},
            {"text": "শক্তি বাড়ায়", "claim_type": "IMPROVEMENT", "evidence": "শক্তি বাড়ায়"},
            {"text": "ওজন বাড়ায়", "claim_type": "IMPROVEMENT", "qualifiers": ["7 দিনে"], "evidence": "7 দিনে দৃশ্যমান পরিবর্তন!\nওজন বাড়ায়"},
            {"text": "শক্তি বাড়ায়", "claim_type": "IMPROVEMENT", "evidence": "শক্তি ও স্ট্যামিনা বাড়ায়"},
            {"text": "GMP সার্টিফাইড", "claim_type": "OTHER", "evidence": "GMP সার্টিফাইড"},
            {"text": "Supports heart", "claim_type": "IMPROVEMENT", "evidence": "Supports heart, brain, eye"},
        ]})


def test_extractor_numbers_claims_after_merging():
    ad: AdExtraction = GroundedExtractor(_Llm(), clean=False).run(ocr_document_from_text("r05", AD)).extraction
    assert [(c.claim_id, c.text_original) for c in ad.claims] == [
        ("c01", "ওজন বাড়ান!"), ("c02", "শক্তি বাড়ায়"), ("c03", "GMP সার্টিফাইড"), ("c04", "Supports heart")]
    assert ad.rejected == []


def test_prompt_says_quality_lines_and_english_text_are_claims():
    for words in ("১০০% ন্যাচারাল", "GMP সার্টিফাইড", "১০০% গ্যারান্টি", "কেমিক্যাল মুক্ত", "English", "list it once"):
        assert words in EXTRACT_SYSTEM
    assert "certifications" not in EXTRACT_SYSTEM.split("- other_text:")[1]
