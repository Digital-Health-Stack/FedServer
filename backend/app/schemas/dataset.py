from pydantic import BaseModel
from datetime import datetime
from typing import Optional,Any, List

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
    benchmark: Optional[dict] = None

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

class SessionLeaderboardEntry(BaseModel):
    session_id: int
    organisation_name: str
    model_name: str
    total_rounds: int
    metric_value: float
    meets_benchmark: Optional[bool]
    # metric_name: str
    # all_metrics: Dict[str, float]
    created_at: Optional[str]
    admin_username: str

class LeaderboardResponse(BaseModel):
    task_name: str
    metric: str
    benchmark: Optional[float]
    created_at: Optional[str]
    sessions: List[SessionLeaderboardEntry]