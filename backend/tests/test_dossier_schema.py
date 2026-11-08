from app.schemas.dossier import ResearchDossier


def test_dossier_rejects_missing_identity():
    dossier = ResearchDossier.model_validate(
        {
            "product_identity": {"identity_status": "not_found"},
            "composition": {"ingredients": [], "composition_status": "not_found"},
        }
    )
    assert dossier.product_identity.identity_status == "not_found"
    assert dossier.medical_disclaimer


def test_verification_status_enum():
    dossier = ResearchDossier.model_validate(
        {
            "product_identity": {"identity_status": "uncertain", "product_name": "Test"},
            "composition": {
                "composition_status": "unverified",
                "ingredients": [
                    {
                        "name": "Zinc",
                        "verification_status": "THIRD_PARTY_REPORTED",
                        "source_url": "https://example.com",
                    }
                ],
            },
        }
    )
    assert dossier.composition.ingredients[0].verification_status == "THIRD_PARTY_REPORTED"
