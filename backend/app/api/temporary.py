from fastapi import APIRouter
import os

temporary_router = APIRouter()

@temporary_router.get('/check')
def check():
    db_url = os.getenv('CHECK_URL')
    return db_url

