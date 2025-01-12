

from pydantic import BaseModel
from typing import Optional

class DatasetBase(BaseModel):
    name: str
    code: str
    description: Optional[str] = None
    source: Optional[str] = None
    
class DatasetCreate(DatasetBase):
    pass