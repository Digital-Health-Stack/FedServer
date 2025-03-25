from pydantic import BaseModel
from typing import Optional

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    datastats: Optional[dict] = None

class DatasetResponse(DatasetCreate):
    dataset_id: int

class TaskCreate(BaseModel):
    dataset_id: int
    task_name: str
    metric: str

class TaskResponse(TaskCreate):
    task_id: int

class BenchmarkCreate(BaseModel):
    task_id: int
    metric_value: float
    training_id: int

class BenchmarkResponse(BenchmarkCreate):
    benchmark_id: int