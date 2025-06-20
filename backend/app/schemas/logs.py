from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FederatedSessionLogResponse(BaseModel):
    id: int
    session_id: int
    message: str
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
