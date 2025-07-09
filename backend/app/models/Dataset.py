from sqlalchemy import (
    Integer,
    String,
    Float,
    ForeignKey,
    JSON,
    TIMESTAMP,
    CheckConstraint,
    Index,
    func,
)
from sqlalchemy.orm import relationship, declarative_base, mapped_column
from datetime import datetime
from models.Base import Base


class RawDataset(Base):
    __tablename__ = "raw_datasets"
    dataset_id = mapped_column(Integer, primary_key=True, index=True)
    filename = mapped_column(String, nullable=False, index=True)
    description = mapped_column(String, nullable=True)
    datastats = mapped_column(JSON)

    def as_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "description": self.description,
            "datastats": self.datastats,
        }


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id = mapped_column(Integer, primary_key=True, index=True)
    filename = mapped_column(String(255), unique=True, nullable=False, index=True)
    description = mapped_column(String, nullable=True)
    datastats = mapped_column(JSON, nullable=True)
    created_at = mapped_column(TIMESTAMP, server_default=func.now(), index=True)
    tasks = relationship("Task", back_populates="dataset", cascade="all, delete")

    def as_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "description": self.description,
            "datastats": self.datastats,
        }


# each dataset will have multiple tasks (decided by server admin), every task will have an associated Metric
class Task(Base):
    __tablename__ = "tasks"
    # Should be auto assigned by the server
    task_id = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True, nullable=False
    )
    dataset_id = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_name = mapped_column(String(255), nullable=False)
    output_column = mapped_column(String(255), nullable=False)
    metric = mapped_column(String(50), nullable=False)
    benchmark = mapped_column(JSON, nullable=True)  # Temporary field
    created_at = mapped_column(TIMESTAMP, server_default=func.now(), index=True)

    dataset = relationship("Dataset", back_populates="tasks")

    __table_args__ = (
        CheckConstraint(
            "metric IN ('mse', 'mae', 'accuracy', 'msle', 'r2', 'logloss', 'auc', 'f1', 'precision', 'recall')"
        ),
    )

    Index("idx_tasks_dataset", dataset_id)
    Index("idx_tasks_metric", metric)

    def as_dict(self):
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "task_name": self.task_name,
            "output_column": self.output_column,
            "metric": self.metric,
            "benchmark": self.benchmark,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
