from pydantic import BaseModel
from datetime import datetime

class FederatedSessionLogResponse(BaseModel):
    id: int
    session_id: int
    message: str
    timestamp: datetime

    class Config:
        orm_mode = True