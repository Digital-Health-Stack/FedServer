############################################################################
############# This file will not be in production ########################## 
############################################################################

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from utility.db import get_db
from models.User import User
from schemas.user import UserCreate
from schemas.dataset import DatasetCreate
from utility.auth import get_password_hash
from dotenv import dotenv_values
from helpers.datasets_crud import create_dataset, create_raw_dataset

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

@confidential_router.post("/create-dataset", status_code=201)
def create_dataset_endpoint(dataset: DatasetCreate, db: Session = Depends(get_db)):
    return create_dataset(db, dataset)

@confidential_router.post("/create-raw-dataset", status_code=201)
def create_raw_dataset_endpoint(dataset: DatasetCreate, db: Session = Depends(get_db)):
    return create_raw_dataset(db, dataset)