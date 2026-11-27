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
from app.services.evidence.selector import Doc, SentenceSelector, sentences
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
