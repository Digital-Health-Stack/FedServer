from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Float, String, event
from sqlalchemy.orm import declared_attr, relationship, Session, with_loader_criteria
from .Base import Base
import os

load_dotenv()

class TimestampMixin:
    @declared_attr
    def transferredAt(cls):
        return Column(DateTime, default=lambda: datetime.now(), nullable=False)

    @declared_attr
    def approvedAt(cls):
        return Column(DateTime, nullable=True)  # NULL means not deleted (soft deletion enabled)

@event.listens_for(Session, "do_orm_execute")
def filter_soft_deleted(execute_state):
    if not execute_state.is_column_load and not execute_state.is_relationship_load:
        if not execute_state.execution_options.get("include_approved", False):
            execute_state.statement = execute_state.statement.options(
                with_loader_criteria(
                    TimestampMixin,
                    lambda cls: cls.approvedAt.is_(None),
                    include_aliases=True
                )
            )

class TrainingDataTransfer(TimestampMixin, Base):
    __tablename__ = "training_data_transfers"

    id = Column(Integer, primary_key=True, index=True)
    training_name = Column(String, nullable=False)  # Name of the training
    num_datapoints = Column(Integer, nullable=False)  # Number of transferred datapoints
    transferred_at = Column(DateTime, default=lambda: datetime.now(), nullable=False)  
    data_path = Column(String, nullable=False)  # Where data is stored
    approved_at = Column(DateTime, nullable=True)  # Becomes non-null when approved


    def as_dict(self):
        return {
            "id": self.id,
            "training_name": self.training_name,
            "num_datapoints": self.num_datapoints,
            "transferred_at": self.transferred_at.isoformat(),
            "data_path": self.data_path,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }
