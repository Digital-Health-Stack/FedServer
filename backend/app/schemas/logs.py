from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FederatedSessionLogResponse(BaseModel):
    id: int
    session_id: int
    message: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True
        validate_by_name = True
