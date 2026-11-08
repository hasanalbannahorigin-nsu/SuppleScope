"""v2 pipeline artifacts: ad extractions and research dossiers

Revision ID: 0002_pipeline_v2
Revises: 0001_initial
Create Date: 2026-11-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0002_pipeline_v2"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from app.models.pipeline_v2 import AdExtractionRecord, ResearchDossierRecord

    bind = op.get_bind()
    AdExtractionRecord.__table__.create(bind=bind, checkfirst=True)
    ResearchDossierRecord.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    op.drop_table("research_dossiers_v2")
    op.drop_table("ad_extractions_v2")
