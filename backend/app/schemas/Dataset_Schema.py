

from pydantic import BaseModel
from typing import Optional,Any,Dict,List

class DatasetBase(BaseModel):
    name: str
    code: str
    description: Optional[str] = None
    source: Optional[str] = None
    columns: Optional[List[Dict[Any,Any]]] = None
    files: Optional[List["DatasetFileBase"]] = None
    
class DatasetCreate(DatasetBase):
    pass



class BenchmarkBase(BaseModel):
    task: str 
    dataset_id: int 
    model_name: str
    benchmark_metric: str
    metrics: Dict[str,Any]
    
class BenchmarkCreate(BenchmarkBase):
    pass

class BenchmarkResponse(BenchmarkBase):
    pass

class DatasetResponse(DatasetBase):
    benchmarks: List[BenchmarkResponse]
    class Config:
        orm_mode = True

class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    columns: Optional[List[Dict[Any,Any]]] = None
    files: Optional[List["DatasetFileBase"]] = None  # Added files attribute
    

class DatasetFileBase(BaseModel):
    name: str
    hdfs_path: str
    is_folder: bool = False
    description: Optional[str] = None
