

from pydantic import BaseModel
from typing import Optional,Any,Dict,List

class DatasetBase(BaseModel):
    name: str
    code: str
    description: Optional[str] = None
    source: Optional[str] = None
    
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
