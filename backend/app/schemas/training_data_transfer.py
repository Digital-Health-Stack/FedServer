from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict


class TransferBase(BaseModel):
    training_name: str
    num_datapoints: int
    data_path: str
    parent_filename: str
    datastats: Optional[Dict] = None
    federated_session_id: int


class TransferCreate(BaseModel):
    training_name: str
    num_datapoints: int
    data_path: str
    parent_filename: str
    datastats: Optional[Dict] = None
    federated_session_id: int


class TransferListItem(BaseModel):
    id: int
    training_name: str
    num_datapoints: int
    data_path: str
    parent_filename: str
    transferredAt: datetime
    approvedAt: Optional[datetime]
    federated_session_id: int


class TransferDetails(BaseModel):
    id: int
    training_name: str
    num_datapoints: int
    data_path: str
    parent_filename: str
    transferredAt: datetime
    approvedAt: Optional[datetime]
    federated_session_id: int
    datastats: Optional[Dict] = None
