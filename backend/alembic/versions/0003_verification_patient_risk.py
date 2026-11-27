"""claim verifier outputs and patient risk reports

Revision ID: 0003_verification_patient_risk
Revises: 0002_pipeline_v2
Create Date: 2026-11-27
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0003_verification_patient_risk"
down_revision: Union[str, None] = "0002_pipeline_v2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from app.models.pipeline_v2 import PatientRiskRecord, VerificationRecord

    bind = op.get_bind()
    VerificationRecord.__table__.create(bind=bind, checkfirst=True)
    PatientRiskRecord.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    op.drop_table("patient_risk_v2")
    op.drop_table("verifications_v2")
