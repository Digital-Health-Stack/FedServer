from pydantic import BaseModel
from datetime import datetime

class FederatedSessionLogResponse(BaseModel):
    id: int
    session_id: int
    message: str
    timestamp: datetime

    # orm_mode was giving warning 
    class Config:
        from_attributes = True