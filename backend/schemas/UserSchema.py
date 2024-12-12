from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    data_url: str
    # email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
    
class RefreshToken(BaseModel):
    refresh_token: str
