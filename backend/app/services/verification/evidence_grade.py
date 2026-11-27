"""Evidence directness and the rule layer of the claim verifier.

Directness grades (GRADE-style indirectness, applied to supplement ads):
  A  finished product, human study, measured the claimed outcome (SUPPORTS/CONTRADICTS)
  B  ingredient, human systematic review / meta-analysis / RCT, measured the claimed outcome
  C  ingredient, other human design, or dose / population clearly different
  D  animal or cell studies only
  E  background (CONTEXT) links only
  F  no linked evidence

Rules (the "evidence-type rule" of the stack):
  - claims with no measurable health outcome -> NOT_ASSESSABLE (only rules give this label)
  - research not done (budget, identity unknown) -> INSUFFICIENT_EVIDENCE, abstained
  - grade D/E/F -> INSUFFICIENT_EVIDENCE (a model may not call it supported or contradicted)
  - ingredient-only evidence (grade B/C) caps the label at PARTIALLY_SUPPORTED
  - SUPPORTED needs grade A with matching qualifiers ("in 7 days", "100%", "permanently")
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.research.claim import EvidenceRelation, MatchLevel
from app.schemas.research.common import EvidenceScope, ResearchStatus, StudyType
from app.schemas.research.verifier import VerdictLabel, VerifierInput

NOT_ASSESSABLE_KINDS = {"general_wellness", "regulatory_or_quality", "other"}
STRONG = {StudyType.SYSTEMATIC_REVIEW, StudyType.META_ANALYSIS, StudyType.RCT}
WEIGHT = {StudyType.META_ANALYSIS: 3.0, StudyType.SYSTEMATIC_REVIEW: 3.0, StudyType.RCT: 2.0,
          StudyType.NON_RANDOMIZED_TRIAL: 1.0, StudyType.OBSERVATIONAL: 1.0, StudyType.CASE_REPORT: 0.5}
GRADES = "ABCDEF"
MODEL_LABELS = [VerdictLabel.SUPPORTED, VerdictLabel.PARTIALLY_SUPPORTED, VerdictLabel.CONTRADICTED,
                VerdictLabel.INSUFFICIENT_EVIDENCE]


@dataclass
class EvidenceProfile:
    grade: str
    supports: float = 0.0  # weighted human direct support
    contradicts: float = 0.0
    product_support_same: bool = False  # grade-A support with matching outcome and qualifiers
    n_direct: int = 0
    n_context: int = 0
    n_preclinical_direct: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def profile(vi: VerifierInput) -> EvidenceProfile:
    ev = {e.evidence_id: e for e in vi.evidence}
    p = EvidenceProfile(grade="F")
    best = "F"
    for link in vi.evidence_links:
        rec = ev.get(link.evidence_id)
        if rec is None or not rec.validation.passed:
            continue
        if link.relation == EvidenceRelation.CONTEXT:
            p.n_context += 1
            best = min(best, "E")
            continue
        p.n_direct += 1
        p.evidence_ids.append(rec.evidence_id)
        human = bool(rec.human_study)
        if not human:
            p.n_preclinical_direct += 1
            best = min(best, "D")
            continue
        w = WEIGHT.get(rec.study_type, 0.5)
        mismatched = link.match.dose == MatchLevel.DIFFERENT or link.match.population == MatchLevel.DIFFERENT
        if rec.scope == EvidenceScope.FINISHED_PRODUCT:
            g = "A"
        elif rec.study_type in STRONG and not mismatched:
            g = "B"
        else:
            g = "C"
        best = min(best, g)
        if link.relation == EvidenceRelation.SUPPORTS:
            p.supports += w
            if g == "A" and link.match.outcome == MatchLevel.SAME and link.match.qualifiers != MatchLevel.DIFFERENT:
                p.product_support_same = True
        else:
            p.contradicts += w
    p.grade = best
    return p


@dataclass
class RuleDecision:
    label: VerdictLabel
    final: bool  # True: the rule decides; False: a model may choose among `allowed`
    allowed: list[VerdictLabel]
    rationale: str
    confidence: float
    abstained: bool = False


def decide(vi: VerifierInput, p: EvidenceProfile | None = None) -> RuleDecision:
    p = p or profile(vi)
    kind = vi.claim_kind
    if kind in NOT_ASSESSABLE_KINDS or (kind is None and vi.claim.claim_type.value == "OTHER" and not vi.evidence_links):
        return RuleDecision(VerdictLabel.NOT_ASSESSABLE, True, [VerdictLabel.NOT_ASSESSABLE],
                            "The claim names no measurable health outcome, so studies cannot be matched to it.", 0.9)
    if not vi.evidence_links and vi.claim_research_status in (ResearchStatus.NOT_SEARCHED, ResearchStatus.IDENTITY_UNCERTAIN):
        why = ("research for this claim was not completed (limit reached)" if vi.claim_research_status == ResearchStatus.NOT_SEARCHED
               else "the product could not be identified and the ad lists no ingredients")
        return RuleDecision(VerdictLabel.INSUFFICIENT_EVIDENCE, True, [VerdictLabel.INSUFFICIENT_EVIDENCE],
                            f"No verdict is possible because {why}.", 0.5, abstained=True)
    if p.grade in "DEF":
        why = {"D": "only animal or cell studies measured this effect",
               "E": "the studies found did not measure the claimed effect itself (background, or a related outcome only)",
               "F": "no study linked to this claim was found"}[p.grade]
        return RuleDecision(VerdictLabel.INSUFFICIENT_EVIDENCE, True, [VerdictLabel.INSUFFICIENT_EVIDENCE],
                            f"Insufficient evidence: {why}.", 0.8 if p.grade == "F" else 0.7)
    allowed = [VerdictLabel.PARTIALLY_SUPPORTED, VerdictLabel.CONTRADICTED, VerdictLabel.INSUFFICIENT_EVIDENCE]
    if p.product_support_same:
        allowed.insert(0, VerdictLabel.SUPPORTED)
    # rule label (also the fallback when no model is configured)
    if p.contradicts > p.supports:
        label, why = VerdictLabel.CONTRADICTED, "human studies that measured the claimed effect mostly found no benefit or harm"
    elif p.supports > 0 and p.product_support_same and p.contradicts == 0:
        label, why = VerdictLabel.SUPPORTED, "a human study of this product measured and supports the claimed effect"
    elif p.supports > 0:
        label = VerdictLabel.PARTIALLY_SUPPORTED
        why = ("human studies support the effect for an ingredient, not for this finished product"
               if p.grade != "A" else "product studies support the effect, but not as strongly as the ad states")
        if p.contradicts:
            why += "; some studies disagree"
    else:
        label, why = VerdictLabel.INSUFFICIENT_EVIDENCE, "no human study supports the claimed effect"
    conf = {"A": 0.8, "B": 0.7, "C": 0.55}.get(p.grade, 0.5)
    return RuleDecision(label, False, allowed, f"Evidence grade {p.grade}: {why}.", conf)


def apply_caps(model_label: VerdictLabel, decision: RuleDecision) -> tuple[VerdictLabel, str | None]:
    """Keep a model's label inside what the evidence type allows."""
    if decision.final:
        return decision.label, None
    if model_label in decision.allowed:
        return model_label, None
    if model_label == VerdictLabel.SUPPORTED:
        return VerdictLabel.PARTIALLY_SUPPORTED, "capped: ingredient-level or mismatched evidence cannot fully support a product claim"
    if model_label == VerdictLabel.NOT_ASSESSABLE:
        return decision.label, "model label NOT_ASSESSABLE ignored: only the rules assign it"
    return decision.label, f"model label {model_label.value} not allowed for evidence grade; rule label used"
