"""Drop raw_datasets table and datasets.datastats column

Revision ID: c7f2a1b3d4e5
Revises: acdaa51b4bfe
Create Date: 2026-04-04

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c7f2a1b3d4e5"
down_revision: Union[str, None] = "acdaa51b4bfe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS raw_datasets CASCADE"))
    op.drop_column("datasets", "datastats")


def downgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("datastats", sa.JSON(), nullable=True),
    )
    op.create_table(
        "raw_datasets",
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("datastats", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("dataset_id"),
    )
    op.create_index(
        op.f("ix_raw_datasets_dataset_id"), "raw_datasets", ["dataset_id"], unique=False
    )
    op.create_index(
        op.f("ix_raw_datasets_filename"), "raw_datasets", ["filename"], unique=False
    )
