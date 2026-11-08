from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class FutureModelInput:
    advertisement_claims: list[str]
    product_identity: dict[str, Any]
    ingredient_composition: dict[str, Any]
    scientific_research_dossier: dict[str, Any]
    patient_context: dict[str, Any] | None


@dataclass
class FutureModelOutput:
    claim_verification: Any
    effectiveness_assessment: Any
    supplement_drug_risk_prediction: Any
    supplement_disease_risk_prediction: Any


class VerificationModel(Protocol):
    """Future custom verification model. Not implemented in this phase."""

    def predict(self, payload: FutureModelInput) -> FutureModelOutput: ...


class NotIntegratedVerificationModel:
    status = "NOT_INTEGRATED"

    def predict(self, payload: FutureModelInput) -> FutureModelOutput:
        raise RuntimeError(
            "Custom AI verification model is not integrated. "
            "Do not generate placeholder predictions."
        )
