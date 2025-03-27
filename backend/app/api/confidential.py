############################################################################
############# This file will not be in production ########################## 
############################################################################

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from utility.db import get_db
from models.User import User
from schemas.user import UserCreate
from utility.auth import get_password_hash
from dotenv import dotenv_values

confidential_router = APIRouter()

@confidential_router.post("/get-env-vars")
def get_env_vars():
    env_vars = dotenv_values(".env")  
    return {"env_variables": env_vars}


@confidential_router.post("/admin-signup", status_code=201)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username Exists")

    # Create new user
    new_user = User(
        username=user.username,
        data_url=user.data_url,
        role='admin',
        hashed_password=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}