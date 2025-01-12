from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from .Base import Base
from sqlalchemy.orm import relationship



class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    code = Column(String, unique = True, index = True, nullable= False)
    description = Column(String, nullable=True)
    source = Column(String)
    benchmarks = relationship("Benchmark", back_populates="dataset")
    