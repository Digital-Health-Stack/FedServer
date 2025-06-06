"""fix metric constraint to lowercase

Revision ID: c21fc2b434e2
Revises: 8d9aeba6444d
Create Date: 2025-04-26 17:28:49.595402

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c21fc2b434e2'
down_revision: Union[str, None] = '8d9aeba6444d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    # Step 1: Drop the old table
    op.drop_table('tasks')

    # Step 2: Create a new table with the updated lowercase CHECK constraint
    op.create_table('tasks',
    sa.Column('task_id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('task_name', sa.String(length=255), nullable=False),
    sa.Column('metric', sa.String(length=50), nullable=False),
    sa.Column('benchmark', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.CheckConstraint("metric IN ('mse', 'mae', 'accuracy', 'msle', 'r2', 'logloss', 'auc', 'f1', 'precision', 'recall')"),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.dataset_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('task_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    # Step 1: Drop the old table
    op.drop_table('tasks')
    op.create_table('tasks',
    sa.Column('task_id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('task_name', sa.String(length=255), nullable=False),
    sa.Column('metric', sa.String(length=50), nullable=False),
    sa.Column('benchmark', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.CheckConstraint("metric IN ('MSE', 'MAE', 'Accuracy', 'MSLE', 'R2', 'LogLoss', 'AUC', 'F1', 'Precision', 'Recall')"),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.dataset_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('task_id')
    )
    
    
