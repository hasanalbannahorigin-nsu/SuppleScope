import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.research import ResearchDossierV2, VerifierInput, VerifierOutput
from app.schemas.research.claim import ClaimResearch
from app.services.verification.handoff import build_verifier_inputs

EXAMPLE = json.loads((Path(__file__).parent / "fixtures" / "example_dossier_v2.json").read_text(encoding="utf-8"))


def test_example_dossier_validates():
    d = ResearchDossierV2.model_validate(EXAMPLE)
    assert d.product.identity.status.value == "PROBABLE"
    assert len(d.claims) == 2


def test_agent_schemas_have_no_verdict_field():
    for model in (ResearchDossierV2, ClaimResearch, VerifierInput):
        text = json.dumps(model.model_json_schema())
        for label in ("PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "NOT_ASSESSABLE", "\"verdict\""):
            assert label not in text, f"{model.__name__} exposes {label}"
    assert "PARTIALLY_SUPPORTED" in json.dumps(VerifierOutput.model_json_schema())


def test_unknown_keys_are_rejected():
    bad = copy.deepcopy(EXAMPLE)
    bad["claims"][0]["verdict"] = "SUPPORTED"
    with pytest.raises(ValidationError):
        ResearchDossierV2.model_validate(bad)


def test_ingredient_evidence_cannot_be_used_as_product_evidence():
    bad = copy.deepcopy(EXAMPLE)
    bad["product"]["finished_product"]["evidence_ids"] = ["ev_001"]  # an INGREDIENT-scope record
    with pytest.raises(ValidationError, match="finished_product cites INGREDIENT evidence"):
        ResearchDossierV2.model_validate(bad)


def test_findings_must_point_at_known_sources():
    bad = copy.deepcopy(EXAMPLE)
    bad["findings"][0]["source_id"] = "src_missing"
    with pytest.raises(ValidationError, match="unknown source"):
        ResearchDossierV2.model_validate(bad)


def test_verifier_inputs_one_per_claim_and_self_contained():
    d = ResearchDossierV2.model_validate(EXAMPLE)
    inputs = build_verifier_inputs(d)
    assert [i.claim.claim_id for i in inputs] == ["c01", "c02"]
    first = inputs[0]
    assert [e.evidence_id for e in first.evidence] == ["ev_001"]
    assert {s.source_id for s in first.sources} == {"src_003", "src_001"}
    assert first.product.identity_status.value == "PROBABLE"
    assert inputs[1].evidence == [] and inputs[1].claim_research_status.value == "NO_RELEVANT_STUDY_FOUND"
    assert any(x.ad_ref == "claims[c02]" for x in inputs[1].discrepancies)
    VerifierInput.model_validate_json(first.model_dump_json())  # round-trips
