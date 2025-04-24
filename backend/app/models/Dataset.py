from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, TIMESTAMP, CheckConstraint, Index, func
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from models.Base import Base

class RawDataset(Base):
    __tablename__ = "raw_datasets"
    dataset_id = Column(Integer, primary_key=True, index=True)
    filename = Column(String,nullable=False, index=True)
    description = Column(String, nullable=True)
    datastats = Column(JSON)

    def as_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "description": self.description,
            "datastats": self.datastats
        }

class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(String, nullable=True)
    datastats = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    tasks = relationship("Task", back_populates="dataset", cascade="all, delete")

    def as_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "description": self.description,
            "datastats": self.datastats
        }

# each dataset will have multiple tasks (decided by server admin), every task will have an associated Metric
class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True)
    task_name = Column(String(255), nullable=False)
    metric = Column(String(50), nullable=False)
    temp_benchmark = Column(JSON, nullable=True)   # Temporary field
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)

    
    dataset = relationship("Dataset", back_populates="tasks")
    benchmarks = relationship("Benchmark", back_populates="task", cascade="all, delete")
    
    
    __table_args__ = (
        CheckConstraint("metric IN ('MSE', 'MAE', 'Accuracy', 'MSLE', 'R2', 'LogLoss', 'AUC', 'F1', 'Precision', 'Recall')"),
    )

    Index("idx_tasks_dataset", dataset_id)
    Index("idx_tasks_metric", metric)

    def as_dict(self):
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "task_name": self.task_name,
            "metric": self.metric,
            "benchmark": self.temp_benchmark
        }

