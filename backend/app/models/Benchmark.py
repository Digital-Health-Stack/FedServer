from sqlalchemy import Column, Integer, ForeignKey, TIMESTAMP, Float, Index, func
from .Base import Base
from sqlalchemy.orm import relationship

    
class Benchmark(Base):
    __tablename__ = "benchmarks"

    benchmark_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    training_id = Column(Integer, ForeignKey("federated_sessions.id"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    task = relationship("Task", back_populates="benchmarks")

    Index("idx_benchmarks_task", task_id)
    Index("idx_benchmarks_time", created_at.desc())

    def as_dict(self):
        return {
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "metric_value": self.metric_value,
            "training_id": self.training_id,
            "created_at": self.created_at
        }