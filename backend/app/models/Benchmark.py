from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from .Base import Base
from sqlalchemy.orm import relationship

    
class Benchmark(Base):
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    task = Column(String)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    model_name = Column(String, nullable=False)
    benchmark_metric = Column(String)  # Name of the primary metric
    metrics = Column(JSON)

    dataset = relationship("Dataset", back_populates="benchmarks")