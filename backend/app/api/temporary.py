from fastapi import APIRouter
import os
from utility.auth import role
from fastapi import Depends
from models.User import User
import subprocess
import sys
from utility.auth import verify_token

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

@temporary_router.get("/run-script")
def run_script():
    # Build relative path to the script
    session_id = str(1)
    client_id = str(1)
    # Clone current environment to include all current variables (like VIRTUAL_ENV)
    env = os.environ.copy()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "utility.test_script", "--session_id", session_id, "--client_id", client_id],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        return {
            "message": "Script executed successfully",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "error": "Script execution failed",
            "stdout": e.stdout,
            "stderr": e.stderr
        }

@temporary_router.get("/check-current_user")
def check_user(token: str):
    current_user = verify_token(token)
    return current_user



