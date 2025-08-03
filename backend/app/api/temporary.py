from fastapi import APIRouter, HTTPException
import os
from utility.auth import role
from fastapi import Depends
from models.User import User
import subprocess
import sys
from utility.auth import verify_token
from utility.db import get_db
from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession
from helpers.federated_services import process_parquet_and_save_xy
from fastapi import Query
from typing import List
from utility.Notification import send_notification_for_new_round

temporary_router = APIRouter()


@temporary_router.get("/test-notification/{session_id}")
async def test_notification(session_id: int):
    await send_notification_for_new_round(
        {
            "session_id": session_id,
            "round_number": 1,
            "metrics_report": {},
        }
    )
    return {"message": "Notification sent"}


@temporary_router.get("/check")
def check():
    return {"message": "Everyone can access it!"}


@temporary_router.get("/check-client")
def check_client(client: User = Depends(role("client"))):
    return {"message": "Only clients can access it!"}


@temporary_router.get("/check-admin")
def check_admin(admin: User = Depends(role("admin"))):
    return {"message": "Only admins can access it!"}


@temporary_router.get("/check-current_user")
def check_user(token: str):
    current_user = verify_token(token)
    return current_user


# API endpoint to call the function
@temporary_router.get("/check-download")
def check_download(
    filename: str,
    session_id: str,
):
    output_column = ["Target"]
    process_parquet_and_save_xy(filename, session_id, output_column)
    return {"message": "Processing complete"}
