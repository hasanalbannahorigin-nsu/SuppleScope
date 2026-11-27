"""One text format for training and inference of the NLI-style claim verifier.

text_a (hypothesis side): the claim in English, its qualifiers and type.
text_b (premise side):    the top-k evidence sentences, each tagged with scope and study design,
                          e.g. "[INGREDIENT|RCT] Jamun seed powder reduced fasting glucose ...".
Public training sets (SciFact, HealthVer, PUBHEALTH) have no scope, so their sentences are tagged
"[STUDY]"; SuppleScope's own labelled claims carry the real tags.
"""
from __future__ import annotations

from app.schemas.research.verifier import VerifierInput

MAX_SENTENCES = 5


def claim_text(text: str, qualifiers: list[str] | None = None, claim_type: str | None = None) -> str:
    parts = [" ".join((text or "").split())]
    if qualifiers:
        parts.append("Qualifiers: " + "; ".join(qualifiers))
    if claim_type:
        parts.append(f"Type: {claim_type}")
    return " | ".join(p for p in parts if p)


def evidence_text(sentences: list[tuple[str, str]], max_sentences: int = MAX_SENTENCES) -> str:
    """sentences: (tag, sentence) pairs, best first."""
    out = [f"[{tag or 'STUDY'}] {' '.join(s.split())}" for tag, s in sentences[:max_sentences] if s and s.strip()]
    return " ".join(out) if out else "[NO EVIDENCE]"


def encode(vi: VerifierInput, max_sentences: int = MAX_SENTENCES) -> tuple[str, str]:
    c = vi.claim
    a = claim_text(c.text_en or c.text_original, c.qualifiers, c.claim_type.value)
    sents = [(s.tag or "", s.sentence) for s in sorted(vi.evidence_sentences, key=lambda x: x.rank)]
    if not sents:  # dossiers made before evidence selection: fall back to the linked passages
        ev = {e.evidence_id: e for e in vi.evidence}
        sents = [(f"{ev[link.evidence_id].scope.value}|{ev[link.evidence_id].study_type.value}" if link.evidence_id in ev else "",
                  link.relevant_passage) for link in vi.evidence_links]
    return a, evidence_text(sents, max_sentences)
