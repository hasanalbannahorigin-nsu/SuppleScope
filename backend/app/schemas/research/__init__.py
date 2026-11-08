"""Versioned hand-off contracts for the SuppleScope research pipeline.

    OCR  ->  AdExtraction  ->  research agent  ->  ResearchDossier  ->  VerifierInput (per claim)
                                                                   ->  custom Verification Model
                                                                   ->  VerifierOutput (stored separately)

Rules encoded here:
- OCR and AdExtraction hold only what is visible/audible in the uploaded ad.
- The research agent writes the ResearchDossier and nothing else. No schema the
  agent writes has a verdict field; verdicts exist only in VerifierOutput.
- Ingredient evidence and finished-product evidence are separate scopes.
"""
from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.claim import ClaimResearch
from app.schemas.research.common import SCHEMA_VERSION
from app.schemas.research.dossier import ResearchDossierV2
from app.schemas.research.ingredient import IngredientResearch
from app.schemas.research.ocr import OcrDocument
from app.schemas.research.product import ProductResearch
from app.schemas.research.verifier import VerifierInput, VerifierOutput

__all__ = [
    "SCHEMA_VERSION",
    "OcrDocument",
    "AdExtraction",
    "ProductResearch",
    "IngredientResearch",
    "ClaimResearch",
    "ResearchDossierV2",
    "VerifierInput",
    "VerifierOutput",
]
