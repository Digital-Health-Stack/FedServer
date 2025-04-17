from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, String, ForeignKey, event, JSON
from sqlalchemy.orm import declared_attr, Session, relationship, with_loader_criteria
from .Base import Base
import os

load_dotenv()

class TimestampMixin:
    @declared_attr
    def transferredAt(cls):
        return Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    @declared_attr
    def approvedAt(cls):
        return Column(DateTime, nullable=True)  # NULL means not approved

# Table for approving data transferred for QPD
class TrainingDataTransfer(TimestampMixin, Base):
    __tablename__ = "training_data_transfers"

    id = Column(Integer, primary_key=True, index=True)
    training_name = Column(String, nullable=False)  # Name of the training
    num_datapoints = Column(Integer, nullable=False)  # Number of transferred datapoints
    data_path = Column(String, nullable=False)  # Where data is stored
    # although parent filename will be in the session information, we need to store it here for easy access
    parent_filename = Column(String, nullable=False)  # Filename of the parent dataset
    datastats = Column(JSON, nullable=True)  # Statistics of the dataset
    federated_session_id = Column(Integer, ForeignKey("federated_sessions.id"), nullable=False, index=True)

    def as_dict(self):
        return {
            "id": self.id,
            "federated_session_id": self.federated_session_id,
            "training_name": self.training_name,
            "num_datapoints": self.num_datapoints,
            "data_path": self.data_path,
            "parent_filename": self.parent_filename,
            "transferredAt": self.transferredAt,
            "approvedAt": self.approvedAt,
            "datastats": self.datastats,
        }

# Fixing soft deletion filtering by applying it on the correct table
@event.listens_for(Session, "do_orm_execute")
def filter_soft_deleted(execute_state):
    if not execute_state.is_column_load and not execute_state.is_relationship_load:
        if not execute_state.execution_options.get("include_approved", False):
            execute_state.statement = execute_state.statement.options(
                with_loader_criteria(
                    TrainingDataTransfer, 
                    lambda cls: cls.approvedAt.is_(None),
                    include_aliases=True
                )
            )


