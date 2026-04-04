from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from constant.enums import FederatedSessionLogTag


class FederatedSessionLogResponse(BaseModel):
    id: int
    session_id: int
    message: str
    created_at: Optional[datetime]
    tag: FederatedSessionLogTag

    class Config:
        from_attributes = True
        validate_by_name = True
