from fastapi import APIRouter
import os
from utility.auth import role
from fastapi import Depends
from models.User import User

temporary_router = APIRouter()

@temporary_router.get('/check')
def check():
    return {"message": "Everyone can access it!"}

@temporary_router.get('/check-client')
def check_client(client: User = Depends(role("client"))):
    return {"message": "Only clients can access it!"}

@temporary_router.get('/check-admin')
def check_admin(admin: User = Depends(role("admin"))):
    return {"message": "Only admins can access it!"}


