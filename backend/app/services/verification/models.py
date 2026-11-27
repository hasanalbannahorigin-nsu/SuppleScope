"""Claim verifiers: fine-tuned transformer ensemble (majority vote), Gemini zero-shot baseline, rules.

Ensemble: one or more fine-tuned sequence-pair classifiers (default recipe: roberta-base and
PubMedBERT, see verifier_training/), each predicting SUPPORTED / PARTIALLY_SUPPORTED / CONTRADICTED /
INSUFFICIENT_EVIDENCE. Labels are combined by majority vote, ties broken by the mean probability
(CliVER, Liu et al. 2024, used a RoBERTa + PubMedBERT + T5 vote; MelAnalyze found RoBERTa F1 0.91).
The evidence-type rule then keeps the label inside what the evidence allows (evidence_grade.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from app.schemas.research.verifier import VerdictLabel, VerifierInput, VerifierOutput
from app.services.verification import evidence_grade as G
from app.services.verification.text_format import encode

logger = logging.getLogger(__name__)
RULES_VERSION = "1.0"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PairClassifier(Protocol):
    name: str
    version: str

    def predict_proba(self, pairs: list[tuple[str, str]]) -> list[dict[VerdictLabel, float]]: ...


class HfPairClassifier:
    """A fine-tuned Hugging Face checkpoint saved by verifier_training/train.py."""

    def __init__(self, path: str | Path, max_length: int = 256, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.path = Path(path)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        id2label = self.model.config.id2label
        self.labels = [VerdictLabel(id2label[i]) for i in range(len(id2label))]
        self.name = self.path.name
        cfg = (self.path / "config.json").read_bytes() if (self.path / "config.json").exists() else b""
        self.version = hashlib.sha256(cfg).hexdigest()[:10]

    def predict_proba(self, pairs):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), 16):
                a, b = zip(*pairs[i:i + 16])
                enc = self.tok(list(a), list(b), truncation="only_second", max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
                out += [dict(zip(self.labels, row)) for row in probs]
        return out


def load_models_from_settings() -> list[PairClassifier]:
    from app.config import get_settings

    s = get_settings()
    models = []
    for d in [x.strip() for x in (s.verifier_model_dirs or "").split(",") if x.strip()]:
        try:
            models.append(HfPairClassifier(d, max_length=s.verifier_max_length))
        except Exception as exc:  # missing torch / transformers or a bad folder
            logger.warning("Verifier model %s not loaded: %s", d, exc)
    return models


def vote(per_model: list[dict[VerdictLabel, float]]) -> tuple[VerdictLabel, dict[VerdictLabel, float]]:
    labels = sorted({lab for p in per_model for lab in p}, key=lambda x: x.value)
    mean = {lab: sum(p.get(lab, 0.0) for p in per_model) / len(per_model) for lab in labels}
    votes = Counter(max(p, key=p.get) for p in per_model)
    top = max(votes.values())
    tied = [lab for lab, n in votes.items() if n == top]
    return max(tied, key=lambda lab: mean[lab]), mean


def _output(vi: VerifierInput, label: VerdictLabel, *, confidence: float | None, rationale: str, model_name: str,
            model_version: str, probs: dict[VerdictLabel, float] | None = None, evidence_ids: list[str] | None = None,
            abstained: bool = False) -> VerifierOutput:
    return VerifierOutput(dossier_id=vi.dossier_id, ad_id=vi.ad_id, claim_id=vi.claim.claim_id, label=label,
                          confidence=None if confidence is None else round(max(0.0, min(1.0, confidence)), 3),
                          label_probabilities={k: round(v, 4) for k, v in probs.items()} if probs else None,
                          abstained=abstained, rationale=rationale[:1500], evidence_ids=evidence_ids or [],
                          model_name=model_name, model_version=model_version, created_at=_now())


class RuleVerifier:
    name = "supplescope-rules"

    def verify(self, inputs: list[VerifierInput]) -> list[VerifierOutput]:
        out = []
        for vi in inputs:
            p = G.profile(vi)
            d = G.decide(vi, p)
            out.append(_output(vi, d.label, confidence=d.confidence, rationale=d.rationale, model_name=self.name,
                               model_version=RULES_VERSION, evidence_ids=p.evidence_ids, abstained=d.abstained))
        return out


class EnsembleVerifier:
    """Majority vote of PairClassifiers, then the evidence-type rule (apply_rules=True)."""

    def __init__(self, models: list[PairClassifier], apply_rules: bool = True) -> None:
        if not models:
            raise ValueError("EnsembleVerifier needs at least one model (VERIFIER_MODEL_DIRS)")
        self.models, self.apply_rules = models, apply_rules
        self.name = "ensemble(" + "+".join(m.name for m in models) + ")" + ("+rules" if apply_rules else "")
        self.version = "-".join(m.version for m in models) + (f"/r{RULES_VERSION}" if apply_rules else "")

    def verify(self, inputs: list[VerifierInput]) -> list[VerifierOutput]:
        if not inputs:
            return []
        pairs = [encode(vi) for vi in inputs]
        per_model = [m.predict_proba(pairs) for m in self.models]
        out = []
        for i, vi in enumerate(inputs):
            label, probs = vote([pm[i] for pm in per_model])
            p = G.profile(vi)
            rationale = f"Model vote: {label.value} (mean probability {probs.get(label, 0):.2f})."
            abstained = False
            if self.apply_rules:
                d = G.decide(vi, p)
                final, note = G.apply_caps(label, d)
                abstained = d.abstained
                rationale = (d.rationale if d.final else rationale + " " + d.rationale) + (f" ({note})" if note else "")
                label = final
            out.append(_output(vi, label, confidence=probs.get(label), rationale=rationale, model_name=self.name,
                               model_version=self.version, probs=probs, evidence_ids=p.evidence_ids, abstained=abstained))
        return out


class _LlmVerdict(BaseModel):
    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


LLM_SYSTEM = """You verify one health claim from a dietary-supplement advertisement against the evidence given.
Labels:
- SUPPORTED: human studies of this finished product measured the claimed effect and support it as stated.
- PARTIALLY_SUPPORTED: evidence supports part of the claim, or only for an ingredient, or less strongly than stated.
- CONTRADICTED: studies that measured the effect found no benefit or found harm.
- INSUFFICIENT_EVIDENCE: no relevant human evidence, or only animal / cell / background evidence.
- NOT_ASSESSABLE: the claim names no measurable health outcome.
Ingredient studies are not studies of the product. Use only the evidence given; cite evidence_ids you used."""


class LlmBaselineVerifier:
    """Zero-shot LLM baseline (e.g. Gemini flash-lite), for comparison with the fine-tuned ensemble."""

    def __init__(self, llm, apply_rules: bool = False) -> None:
        self.llm, self.apply_rules = llm, apply_rules
        self.name = f"llm-zero-shot({getattr(llm, 'model', 'llm')})" + ("+rules" if apply_rules else "")

    def verify(self, inputs: list[VerifierInput]) -> list[VerifierOutput]:
        out = []
        for vi in inputs:
            a, b = encode(vi)
            ev = [{"evidence_id": e.evidence_id, "scope": e.scope.value, "study_type": e.study_type.value,
                   "human_study": e.human_study, "passage": e.passage[:800]} for e in vi.evidence]
            user = json.dumps({"claim": a, "evidence_sentences": b, "evidence_records": ev,
                               "product_identity": vi.product.identity_status.value}, ensure_ascii=False, indent=1)
            try:
                r = self.llm.parse(LLM_SYSTEM, user, _LlmVerdict)
            except Exception as exc:
                out.append(_output(vi, VerdictLabel.INSUFFICIENT_EVIDENCE, confidence=None, abstained=True,
                                   rationale=f"LLM call failed: {str(exc)[:200]}", model_name=self.name, model_version="1"))
                continue
            label, rationale = r.label, r.rationale
            if self.apply_rules:
                final, note = G.apply_caps(label, G.decide(vi))
                label, rationale = final, rationale + (f" ({note})" if note else "")
            known = {e.evidence_id for e in vi.evidence}
            out.append(_output(vi, label, confidence=r.confidence, rationale=rationale, model_name=self.name, model_version="1",
                               evidence_ids=[e for e in r.evidence_ids if e in known]))
        return out


def verifier_from_settings(mode: str | None = None, *, llm=None, models: list[PairClassifier] | None = None):
    """auto: ensemble+rules when VERIFIER_MODEL_DIRS loads, else rules. Others: rules, ensemble, ensemble+rules,
    llm_baseline, llm_baseline+rules."""
    from app.config import get_settings

    mode = (mode or get_settings().verifier_mode or "auto").lower()
    if mode == "rules":
        return RuleVerifier()
    if mode.startswith("llm_baseline"):
        if llm is None:
            from app.services.llm.client import llm_from_settings

            llm = llm_from_settings()
        if llm is None:
            raise RuntimeError("llm_baseline needs LLM_API_KEY")
        return LlmBaselineVerifier(llm, apply_rules=mode.endswith("+rules"))
    models = models if models is not None else load_models_from_settings()
    if mode in ("ensemble", "ensemble+rules"):
        if not models:
            raise RuntimeError("No verifier model loaded: set VERIFIER_MODEL_DIRS to the trained model folders")
        return EnsembleVerifier(models, apply_rules=mode == "ensemble+rules")
    return EnsembleVerifier(models, apply_rules=True) if models else RuleVerifier()
