from sqlalchemy import Column, Integer, String, ForeignKey, JSON, Text,Boolean
from .Base import Base
from sqlalchemy.orm import relationship


class DatasetFile(Base):
    __tablename__ = "dataset_files"
    id = Column(Integer, primary_key=True, index=True)
    dataset_code = Column(String, ForeignKey("datasets.code"),nullable=False)
    name = Column(String, nullable = False)
    hdfs_path = Column(Text, nullable=False)
    is_folder = Column(Boolean, default=False)
    description = Column(Text, nullable=True)
    
    dataset = relationship("Dataset",back_populates="files")
    
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    code = Column(String, unique = True, index = True, nullable= False)
    description = Column(String, nullable=True)
    source = Column(String)
    columns = Column(JSON, nullable=True)
    benchmarks = relationship("Benchmark", back_populates="dataset")
    files = relationship("DatasetFile",back_populates="dataset")

    
    