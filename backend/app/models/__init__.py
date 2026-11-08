from app.models.advertisement import Advertisement
from app.models.extraction import ExtractedClaim, ExtractedProduct, MentionedIngredient
from app.models.future import FutureModelConfiguration
from app.models.media import ExtractedFrame, OcrResult, SpeechTranscript, UploadedFile
from app.models.pipeline_v2 import AdExtractionRecord, ResearchDossierRecord
from app.models.research import (
    DiseaseRiskResearch,
    EvidencePassage,
    IngredientResearch,
    InteractionResearch,
    PatientContext,
    ProductComposition,
    ProductCompositionIngredient,
    ProductIdentification,
    ResearchJob,
    ResearchReport,
    ResearchSource,
    ScientificPaper,
)
from app.models.user import User

__all__ = [
    "Advertisement",
    "User",
    "UploadedFile",
    "ExtractedFrame",
    "OcrResult",
    "SpeechTranscript",
    "ExtractedProduct",
    "ExtractedClaim",
    "MentionedIngredient",
    "ProductIdentification",
    "ProductComposition",
    "ProductCompositionIngredient",
    "ResearchJob",
    "ResearchSource",
    "ScientificPaper",
    "EvidencePassage",
    "IngredientResearch",
    "InteractionResearch",
    "DiseaseRiskResearch",
    "PatientContext",
    "ResearchReport",
    "FutureModelConfiguration",
    "AdExtractionRecord",
    "ResearchDossierRecord",
]
