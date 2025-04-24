from pydantic import BaseModel
from datetime import datetime
from typing import Optional,Any

class DatasetCreate(BaseModel):
    filename: str
    description: Optional[str] = None
    datastats: Optional[dict] = None

class DatasetResponse(DatasetCreate):
    dataset_id: int

class DatasetUpdate(BaseModel):
    dataset_id: int
    filename: str
    description: Optional[str] = None

class TaskCreate(BaseModel):
    dataset_id: int
    task_name: str
    metric: str
    benchmark: Optional[Any] = None

class TaskResponse(TaskCreate):
    task_id: int

class RawDatasetListResponse(BaseModel):
    dataset_id: int
    filename: str
    description: Optional[str] = None

class DatasetListResponse(BaseModel):
    dataset_id: int
    filename: str
    description: Optional[str] = None
    created_at: datetime

class Operation(BaseModel):
    column: str
    operation: str