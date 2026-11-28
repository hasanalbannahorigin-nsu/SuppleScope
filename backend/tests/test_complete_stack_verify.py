"""Completed stack, decision side: evidence selection, claim verifier (rules, ensemble vote, LLM baseline),
patient risk, metrics and the API endpoints. Offline: fake models stand in for the fine-tuned checkpoints."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.schemas.research.ad_extraction import AdClaim, ClaimType, TextSpan
from app.schemas.research.common import ResearchStatus
from app.schemas.research.patient_risk import PatientProfile, RiskLevel
from app.schemas.research.verifier import VerdictLabel
from app.services.evaluation.metrics import cohen_kappa, fleiss_kappa, report
from app.services.evidence.selector import Doc, SentenceSelector, labelled_sentences, sentence_role, sentences
from app.services.patient_risk.engine import assess, normalise_conditions, normalise_medicine, profile_from_context
from app.services.verification import evidence_grade as G
from app.services.verification.handoff import build_verifier_inputs
from app.services.verification.models import (
    EnsembleVerifier,
    LlmBaselineVerifier,
    RuleVerifier,
    verifier_from_settings,
    vote,
)
from app.services.verification.text_format import encode

from tests.test_research_agent_v2 import _full_run

S, P, C, I = (VerdictLabel.SUPPORTED, VerdictLabel.PARTIALLY_SUPPORTED, VerdictLabel.CONTRADICTED,
              VerdictLabel.INSUFFICIENT_EVIDENCE)


@pytest.fixture(scope="module")
def dossier():
    return _full_run()


@pytest.fixture(scope="module")
def inputs(dossier):
    return {vi.claim.claim_id: vi for vi in build_verifier_inputs(dossier)}


# ---- evidence selection ------------------------------------------------------------------------------------

DOCS = [Doc("ev1", "s1", "Background: Jamun is a fruit. Results: Jamun seed powder reduced fasting blood glucose in adults "
                        "with type 2 diabetes. Mild nausea occurred in two people.", "INGREDIENT|RCT"),
        Doc("ev2", "s2", "Gymnema extract improved lipid profile in rats fed a high-fat diet for six weeks.", "INGREDIENT|ANIMAL")]


def test_bm25_selects_claim_sentences():
    assert sentences(DOCS[0].text)[0].startswith("Jamun seed powder reduced")  # "Results:" stripped, short one dropped
    sel = SentenceSelector(mode="bm25", top_k=2)
    out = sel.select("helps control blood glucose in diabetes", DOCS)
    assert out[0].sentence.startswith("Jamun seed powder reduced") and out[0].retriever == "bm25"
    assert out[0].tag == "INGREDIENT|RCT" and [x.rank for x in out] == list(range(1, len(out) + 1))
    assert SentenceSelector(mode="bm25").select("", DOCS) == []


class _Enc:
    def encode(self, texts, normalize_embeddings=True):
        import math
        vecs = []
        for t in texts:
            v = [float("glucose" in t.lower()), float("rats" in t.lower()), 0.1]
            n = math.sqrt(sum(x * x for x in v))
            vecs.append([x / n for x in v])
        return vecs


class _Rerank:
    def predict(self, pairs):
        return [2.0 if "nausea" in s.lower() else 0.0 for _, s in pairs]


def test_neural_path_reranks_with_cross_encoder():
    pytest.importorskip("numpy")
    sel = SentenceSelector(mode="auto", top_k=3, encoder=_Enc(), reranker=_Rerank())
    out = sel.select("blood glucose", DOCS)
    assert out[0].sentence.startswith("Mild nausea") and out[0].retriever == "bm25+embedding+cross-encoder"


TITLE = ("The efficacy of folk medicines in the management of type 2 diabetes mellitus: results of a randomized "
         "controlled trial of Syzygium cumini (L.) Skeels.")


def test_sentence_roles_from_cues_and_section_labels():
    cases = {
        "With the global incidence of diabetes mellitus surpassing 366 million, interest in herbal remedies has surged.": "background",
        TITLE: "title",
        "To investigate whether a tea prepared from leaves of jambolan has an antihyperglycaemic effect in diabetes.": "aim",
        "Given its lower cost, rigorous randomized controlled trials should be conducted with fenugreek in diabetes.": "advice",
        "Systematic reviews indicate that Ashwagandha effectively restores blood glucose and HbA1c levels.": "conclusion",
        "No differences were found between the jambolan and placebo groups in fasting glucose.": "result",
        "Fenugreek reduced fasting blood glucose by 0.96 mmol/L (95% CI 0.5 to 1.4) compared with control.": "result",
        "Forty-eight patients were randomized to receive jambolan tea or placebo for 28 days.": "methods",
    }
    for text, role in cases.items():
        assert sentence_role(text, None, TITLE) == role, text
    structured = (TITLE + "\nBACKGROUND: Diabetes is common in adults all over the world.\nRESULTS: Glucose fell more in the tea "
                  "group than with placebo. Nausea occurred in two people overall.\nCONCLUSIONS: Larger trials are needed "
                  "before any use in patients.\nTRIAL REGISTRATION: ClinicalTrials.gov NCT01234567 registered.")
    got = labelled_sentences(structured)
    assert [sec for _, sec in got] == [None, "background", "result", "result", "conclusion", "other"]
    assert sentence_role(got[4][0], got[4][1]) == "advice"  # "trials are needed" in the conclusions is advice
    html = labelled_sentences("<h4>Aim of the study</h4>We aimed to test <i>Trigonella</i> seeds in adults with diabetes.")
    assert html == [("We aimed to test Trigonella seeds in adults with diabetes.", "aim")]


JAMBOLAN = Doc("ev3", "s3", TITLE + "\nTo investigate whether a tea prepared from leaves of jambolan lowers blood glucose in "
                                  "type 2 diabetes mellitus. Fasting blood glucose in type 2 diabetes did not differ between "
                                  "jambolan tea and placebo after 28 days (p = 0.6). Rigorous trials of jambolan in type 2 "
                                  "diabetes should be conducted.", "INGREDIENT|RCT", title=TITLE)
FENUGREEK = Doc("ev4", "s4", "With the global incidence of diabetes mellitus surpassing 366 million, interest in fenugreek "
                             "has surged. Fenugreek lowered fasting blood glucose in type 2 diabetes mellitus by 15% versus "
                             "placebo.", "INGREDIENT|META_ANALYSIS")


def test_findings_rank_above_title_background_aim_and_advice():
    query = "diabetes is cured type 2 diabetes mellitus blood glucose"
    out = SentenceSelector(mode="bm25", top_k=4).select(query, [JAMBOLAN, FENUGREEK])
    assert [x.role for x in out[:2]] == ["result", "result"]
    assert {x.doc_id for x in out[:2]} == {"ev3", "ev4"}  # one finding from each study before the rest
    assert out[0].sentence.startswith(("Fasting blood glucose", "Fenugreek lowered"))
    assert all(x.role in ("title", "aim", "advice", "background") for x in out[2:])
    plain = SentenceSelector(mode="bm25", top_k=4, role_bonus=0.0).select(query, [JAMBOLAN, FENUGREEK])
    assert [x.role for x in plain[:2]] == ["result", "aim"]  # relevance alone ranks the aim above a finding


ASHWA = [
    "With the global incidence of diabetes mellitus (DM) surpassing 366 million, interest in herbal remedies like "
    "Ashwagandha has surged.",
    "Systematic reviews indicate that Ashwagandha effectively restores altered blood glucose and glycosylated haemoglobin "
    "(HbA1c) levels without significant safety concerns.",
    "W. somnifera treatment in DM significantly restored the altered levels of blood glucose (experimental data; mean "
    "difference, -196.27; 95% confidence interval [-220.96, -171.58]; p < .00001).",
    "Withania somnifera Dunal, also known as Indian ginseng, has been in use since ancient times in the management of "
    "diabetes mellitus (DM).",
    "This systematic review and meta-analysis evaluated its effects on fasting blood glucose (FBG) and glycated hemoglobin.",
]


def test_roles_of_real_ashwagandha_sentences():
    assert [sentence_role(x) for x in ASHWA] == ["background", "conclusion", "result", "background", "aim"]
    # "significant potential" is not a measured result; "increasing amount of evidence" is background
    assert sentence_role("According to an increasing amount of scientific and clinical evidence, this fruit shows "
                         "significant potential for application in oral health management.") == "background"


def test_one_outlying_cross_encoder_score_does_not_bury_findings():
    pytest.importorskip("numpy")
    scores = dict(zip(ASHWA, [9.0, 2.0, 1.5, 1.2, 1.0]))  # the background line scores far above the rest

    class _Flat:
        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0]] * len(texts)

    class _Outlier:
        def predict(self, pairs):
            return [scores[s] for _, s in pairs]

    docs = [Doc("n1", "s1", " ".join(ASHWA[:2]), "INGREDIENT|NARRATIVE_REVIEW"),
            Doc("m1", "s2", " ".join(ASHWA[2:]), "INGREDIENT|META_ANALYSIS")]
    out = SentenceSelector(encoder=_Flat(), reranker=_Outlier()).select("diabetes is cured blood glucose", docs)
    assert [x.role for x in out[:2]] == ["conclusion", "result"]
    assert out[2].role == "background" and out[-1].role == "aim"


def test_no_split_after_plant_name_authorities():
    text = ("We compared Lycium barbarum L. (LJT) and Rehmannia glutinosa Libosch. (RGL) in diabetic mice for 4 weeks. "
            "Rehmannia glutinosa (Gaertn.) DC. (RG) lowered blood glucose in diabetic mice. The mice were also given "
            "vitamin C. They then rested for a week before the tests.")
    got = sentences(text)
    assert got[0].startswith("We compared Lycium barbarum L. (LJT) and Rehmannia glutinosa Libosch. (RGL)")
    assert got[1].startswith("Rehmannia glutinosa (Gaertn.) DC. (RG) lowered")
    assert not any(s.startswith("(") for s in got)
    assert got[2:] == ["The mice were also given vitamin C.", "They then rested for a week before the tests."]


def test_background_lines_under_a_results_label():
    text = ("RESULTS: Rehmannia glutinosa has been widely used in the treatment of diabetes since ancient times. The active "
            "ingredients studied most are Rehmannia polysaccharide and oligosaccharide. Fasting glucose fell by 20% in the "
            "treated mice. Nausea occurred in two of the forty volunteers.")
    roles = [sentence_role(s, sec) for s, sec in labelled_sentences(text)]
    assert roles == ["background", "other", "result", "result"]
    assert sentence_role("Prepared Rehmannia glutinosa is an anti-diabetic traditional Chinese medicine whose monomers "
                         "exhibit antioxidant effects.") == "background"


def test_at_most_two_sentences_per_study_before_others():
    out = SentenceSelector(mode="bm25", top_k=3).select("type 2 diabetes jambolan glucose", [JAMBOLAN, FENUGREEK])
    assert [x.doc_id for x in out].count("ev3") <= 2 and "ev4" in {x.doc_id for x in out}


def test_dossier_carries_top_sentences(inputs):
    c01 = inputs["c01"]
    assert c01.evidence_sentences and c01.evidence_sentences[0].retriever == "bm25"
    assert c01.claim_kind == "health_effect"
    a, b = encode(c01)
    assert a.startswith("helps control sugar level") and "Type: TREATMENT" in a
    assert b.startswith("[INGREDIENT|") and "reduced fasting blood glucose" in b


# ---- rules and evidence grade --------------------------------------------------------------------------------

def test_ingredient_only_support_is_partial(inputs):
    p = G.profile(inputs["c01"])
    assert p.grade == "B" and p.supports > 0 and not p.product_support_same
    d = G.decide(inputs["c01"], p)
    assert d.label == P and not d.final and S not in d.allowed
    out = RuleVerifier().verify([inputs["c01"], inputs["c02"]])
    assert [o.label for o in out] == [P, I]
    assert "Evidence grade B" in out[0].rationale and out[0].evidence_ids
    assert "background" in out[1].rationale  # c02 has CONTEXT links only


def _claim(kind_type=ClaimType.OTHER):
    return AdClaim(claim_id="c09", text_original="সেরা মানের পণ্য", text_en="best quality product", claim_type=kind_type,
                   span=TextSpan(text="সেরা মানের পণ্য", segment_id="seg01"))


def test_not_assessable_and_incomplete_research(inputs):
    vi = inputs["c01"].model_copy(update={"claim": _claim(), "claim_kind": "general_wellness", "evidence_links": []})
    assert RuleVerifier().verify([vi])[0].label == VerdictLabel.NOT_ASSESSABLE
    vi2 = inputs["c01"].model_copy(update={"evidence_links": [], "claim_research_status": ResearchStatus.NOT_SEARCHED})
    o = RuleVerifier().verify([vi2])[0]
    assert o.label == I and o.abstained


class _Model:
    def __init__(self, name, label, p=0.8):
        self.name, self.version, self.label, self.p = name, "v1", label, p

    def predict_proba(self, pairs):
        rest = (1 - self.p) / 3
        return [{lab: (self.p if lab == self.label else rest) for lab in G.MODEL_LABELS} for _ in pairs]


def test_majority_vote_and_tie_break():
    lab, mean = vote([{S: 0.6, I: 0.4}, {S: 0.7, I: 0.3}, {I: 0.9, S: 0.1}])
    assert lab == S and round(mean[S], 3) == 0.467
    lab, _ = vote([{S: 0.51, I: 0.49}, {I: 0.95, S: 0.05}])  # 1-1 tie -> higher mean probability
    assert lab == I


def test_ensemble_vote_is_capped_by_evidence_type(inputs):
    ens = EnsembleVerifier([_Model("roberta-ft", S), _Model("pubmedbert-ft", S), _Model("t5", C)])
    out = ens.verify([inputs["c01"], inputs["c02"]])
    assert out[0].label == P and "capped" in out[0].rationale  # SUPPORTED needs product-level evidence
    assert out[1].label == I  # context only: the rule decides
    assert out[0].label_probabilities and ens.name == "ensemble(roberta-ft+pubmedbert-ft+t5)+rules"
    raw = EnsembleVerifier([_Model("a", S)], apply_rules=False).verify([inputs["c01"]])[0]
    assert raw.label == S


def test_confidence_comes_from_whoever_chose_the_label(inputs):
    # laptop 2026-11-28: an abstained claim showed "confidence 3%", the models' probability for a label the rule set
    ens = EnsembleVerifier([_Model("a", S, p=0.97), _Model("b", S, p=0.97)])
    stopped = inputs["c01"].model_copy(update={"evidence_links": [], "claim_research_status": ResearchStatus.NOT_SEARCHED})
    capped, context, abstained = ens.verify([inputs["c01"], inputs["c02"], stopped])
    assert abstained.abstained and abstained.confidence is None and abstained.label_probabilities
    assert capped.label == P and capped.confidence == G.decide(inputs["c01"]).confidence  # rule cap, rule confidence
    assert context.label == I and context.confidence == 0.7
    agree = EnsembleVerifier([_Model("a", P, p=0.6)]).verify([inputs["c01"]])[0]
    assert agree.label == P and agree.confidence == 0.6  # the models' own label keeps their probability
    assert RuleVerifier().verify([stopped])[0].confidence is None


def test_overruled_model_vote_is_shown(inputs):
    # laptop 2026-11-28, "Diabetes is cured": the rule decided and the models' vote was not visible
    context = EnsembleVerifier([_Model("a", S), _Model("b", S)]).verify([inputs["c02"]])[0]
    assert context.label == I
    assert "did not measure the claimed effect itself" in context.rationale
    assert "The models voted SUPPORTED (mean probability 0.80); the evidence rule overrules that." in context.rationale
    agree = EnsembleVerifier([_Model("a", I)]).verify([inputs["c02"]])[0]
    assert "voted" not in agree.rationale


def test_literature_search_sentence_is_methods():
    from app.services.evidence.selector import sentence_role

    assert sentence_role("An electronic literature search was carried out in the PubMed, Google Scholar, Scopus, and "
                         "Cochrane Library databases through 18 November 2022 to find trials that assessed fasting "
                         "blood glucose.") == "methods"


class _Llm:
    model = "fake-gemini"

    def parse(self, system, user, schema, *, counter=None):
        return schema(label="SUPPORTED", confidence=0.9, rationale="Studies show it works.", evidence_ids=["ev_001", "bogus"])


def test_llm_baseline_and_settings(inputs):
    raw = LlmBaselineVerifier(_Llm()).verify([inputs["c01"]])[0]
    assert raw.label == S and "bogus" not in raw.evidence_ids
    capped = LlmBaselineVerifier(_Llm(), apply_rules=True).verify([inputs["c01"]])[0]
    assert capped.label == P
    assert isinstance(verifier_from_settings("auto", models=[]), RuleVerifier)
    assert isinstance(verifier_from_settings("auto", models=[_Model("m", S)]), EnsembleVerifier)
    with pytest.raises(RuntimeError):
        verifier_from_settings("ensemble", models=[])


# ---- patient risk ------------------------------------------------------------------------------------------------

def test_medicine_and_condition_normalisation():
    f = normalise_medicine("Comet 500 mg tablet")
    assert f.name == "metformin" and "antidiabetic" in f.classes and f.normalised_by == "brand_table"
    assert normalise_medicine("warfarin 5mg").classes == ["anticoagulant"]
    assert normalise_medicine("Xyzzol", rxnorm=lambda t: "sertraline").normalised_by == "rxnorm"
    assert normalise_medicine("Xyzzol").normalised_by == "none"
    names = {c.name for c in normalise_conditions(["আমার ডায়াবেটিস আছে", "গর্ভবতী", "লো প্রেশার"])}
    assert names == {"diabetes", "pregnancy", "low blood pressure"}
    prof = profile_from_context(40, "diabetes, kidney", "Comet 500; Ecosprin 75", None)
    assert prof.medications == ["Comet 500", "Ecosprin 75"]


def test_patient_risk_flags_glucose_lowering_with_metformin(dossier):
    rep = assess(dossier, PatientProfile(age=52, medications=["Comet 500", "Xyzzol"], conditions=["type 2 diabetes"]))
    assert rep.overall == RiskLevel.MEDIUM
    r05 = [i for i in rep.items if i.rule_id == "R05"]
    assert {i.ingredient for i in r05} >= {"Syzygium cumini seed extract", "Gymnema sylvestre leaf extract"}
    assert all("metformin" in i.factor and i.level == RiskLevel.MEDIUM for i in r05)
    assert any("Xyzzol" in u for u in rep.unknowns)
    assert "pharmacist" in rep.disclaimer
    none = assess(dossier, PatientProfile())
    assert none.overall == RiskLevel.UNKNOWN and not none.items
    ok = assess(dossier, PatientProfile(medications=["Napa"]))
    assert ok.overall == RiskLevel.LOW and not ok.items


def test_patient_risk_uses_dossier_findings(dossier):
    from app.schemas.research.ingredient import DrugInteraction
    ing = dossier.ingredients[0]
    safety = ing.safety.model_copy(update={"drug_interactions": [DrugInteraction(
        drug_or_class="warfarin", description="May increase the effect of blood thinners.", finding_ids=[])]})
    d2 = dossier.model_copy(update={"ingredients": [ing.model_copy(update={"safety": safety}), *dossier.ingredients[1:]]})
    rep = assess(d2, PatientProfile(medications=["warfarin"]))
    item = next(i for i in rep.items if i.basis == "dossier_finding")
    assert item.factor == "warfarin" and item.level == RiskLevel.MEDIUM


def test_patient_risk_does_not_report_body_parts_as_unresearched(dossier):
    extra = dossier.ingredients[0].model_copy(update={
        "ingredient_id": "ing_ad_99", "name": "pancreatic beta cell", "standardized_name": "pancreatic beta cell",
        "safety": dossier.ingredients[0].safety.model_copy(update={"status": ResearchStatus.NOT_SEARCHED})})
    d2 = dossier.model_copy(update={"ingredients": [*dossier.ingredients, extra]})
    rep = assess(d2, PatientProfile(medications=["Comet 500"]))
    assert not any("pancreatic beta cell" in u for u in rep.unknowns)


def test_patient_risk_ignores_sentences_that_only_name_the_disease(dossier):
    # NCCIH pages say "fenugreek is promoted for diabetes"; that is not a caution for a diabetic patient.
    from app.schemas.research.ingredient import DiseaseCaution
    ing = dossier.ingredients[0]
    safety = ing.safety.model_copy(update={"disease_cautions": [
        DiseaseCaution(condition="diabetes", description="Today, fenugreek seed is promoted as a dietary supplement for "
                       "diabetes, menstrual cramps, and many other conditions.", finding_ids=[]),
        DiseaseCaution(condition="diabetes", description="There is not enough high-quality evidence to determine whether "
                       "fenugreek is useful for diabetes or menstrual cramps.", finding_ids=[]),
        DiseaseCaution(condition="kidney disease", description="Bethesda, MD: National Institute of Diabetes and "
                       "Digestive and Kidney Diseases; 2019.", finding_ids=[]),
        DiseaseCaution(condition="pregnancy", description="Fenugreek may be unsafe for use during pregnancy.",
                       finding_ids=[])]})
    d2 = dossier.model_copy(update={"ingredients": [ing.model_copy(update={"safety": safety}), *dossier.ingredients[1:]]})
    found = [i for i in assess(d2, PatientProfile(conditions=["type 2 diabetes", "pregnant", "kidney disease"])).items
             if i.basis == "dossier_finding"]
    assert [i.factor for i in found] == ["pregnancy"]


# ---- metrics --------------------------------------------------------------------------------------------------------

def test_metrics_match_hand_computed_values():
    gold = ["S", "S", "P", "I", "I", "C"]
    pred = ["S", "P", "P", "I", "S", "C"]
    r = report(gold, pred)
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["per_class"]["S"]["precision"] == pytest.approx(0.5) and r["per_class"]["S"]["recall"] == pytest.approx(0.5)
    assert r["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 2 / 3 + 1.0) / 4)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == pytest.approx(0.0)
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == 1.0
    try:
        from sklearn.metrics import cohen_kappa_score, f1_score
    except ImportError:
        return
    assert r["macro_f1"] == pytest.approx(f1_score(gold, pred, average="macro"))
    assert r["cohen_kappa"] == pytest.approx(cohen_kappa_score(gold, pred))


# ---- API ---------------------------------------------------------------------------------------------------------------

def test_verify_and_patient_risk_endpoints(dossier):
    from app.api.routes import advertisements as A
    from app.models import Advertisement, PatientContext, ResearchDossierRecord, User

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    user = User(email="t@example.com", hashed_password="x", full_name="T")
    db.add(user)
    db.flush()
    ad = Advertisement(user_id=user.id, original_filename="ad.png", media_type="image", content_type="image/png")
    db.add(ad)
    db.flush()
    db.add(ResearchDossierRecord(advertisement_id=ad.id, dossier_id=dossier.dossier_id, schema_version=dossier.schema_version,
                                 payload=dossier.model_dump(mode="json")))
    db.commit()
    body = A.verify_claims(ad.id, A.VerifyIn(mode="rules"), user=user, db=db)
    assert body["verifier"] == "supplescope-rules" and [o["label"] for o in body["outputs"]] == ["PARTIALLY_SUPPORTED",
                                                                                                "INSUFFICIENT_EVIDENCE"]
    assert A.get_verification(ad.id, user=user, db=db)["dossier_id"] == dossier.dossier_id
    with pytest.raises(Exception):
        A.patient_risk(ad.id, user=user, db=db)  # no patient context yet
    db.add(PatientContext(advertisement_id=ad.id, age=50, diseases="ডায়াবেটিস", medications="Comet 500"))
    db.commit()
    risk = A.patient_risk(ad.id, user=user, db=db)
    assert risk["overall"] == "MEDIUM" and A.get_patient_risk(ad.id, user=user, db=db)["overall"] == "MEDIUM"


def test_roles_of_d_care_evidence_sentences():
    # D CARE re-test on the laptop, 2026-11-28
    from app.services.evidence.selector import sentence_role

    cases = {
        "There were no differences in age, sex, or glycated hemoglobin (HbA1c) levels between the bitter melon extract "
        "and placebo groups.": "methods",
        "The Azadirachta indica (neem) is an evergreen tree belonging to the family of Meliaceae.": "background",
        "A related species Tinospora cordifolia is used in Ayurveda for treating a large spectrum of diseases.": "background",
        "While, Tinospora cordifolia is widely recognized as a significant herb in the Indian System of Medicines (ISM) due "
        "to its bioactive components and has been used in the treatment of diabetes, cancer, fever and many more.": "background",
        "Our present study in this review encompasses 1.) Phytochemistry, traditional uses and pharmacological potential "
        "of T. cordifolia as well as other Indian Tinospora species.": "aim",
        "In this paper, an attempt has been made to review the effects of various extracts from Gymnema sylvestre in the "
        "regulation of carbohydrate and lipid metabolism in both animal and clinical studies.": "aim",
        "Relevant clinical trials are necessary to validate the mentioned effects, safety, and optimum dose of this herbal "
        "medicine.": "advice",
        # findings stay findings
        "Our meta-analysis of 10 studies with 419 participants showed that GS supplementation significantly reduces fasting "
        "blood glucose (SMD 1.57 mg/dl, 95% CI 2.22 to -0.93, p < .0001) compared to baseline.": "result",
        "Fasting blood glucose levels decreased significantly with glyburide and did not change with S. cumini tea or "
        "placebo.": "result",
        "No significant differences in HbA1c were found between the extract and placebo groups after 12 weeks.": "result",
    }
    assert {s: sentence_role(s) for s in cases} == cases
    assert sentence_role(next(iter(cases)), section="result") == "methods"  # also under a RESULTS label


def test_reason_says_when_models_choose_another_allowed_label(inputs):
    out = EnsembleVerifier([_Model("a", I), _Model("b", I)]).verify([inputs["c01"]])[0]
    assert out.label == I  # allowed for grade B evidence, so the models' label stands
    assert "The rules alone would say PARTIALLY_SUPPORTED" in out.rationale
    same = EnsembleVerifier([_Model("a", P)]).verify([inputs["c01"]])[0]
    assert "rules alone" not in same.rationale
