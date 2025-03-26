from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from utility.db import get_db
from datetime import datetime
from schemas.user import RefreshToken, UserCreate, UserLogin
from models.User import User
from utility.auth import get_password_hash, verify_password, decode_refresh_token, create_tokens, get_current_user

user_router = APIRouter()

# remove this end-point later
@user_router.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

# Signup route
@user_router.post("/signup", status_code=201)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username Exists")

    # Create new user
    new_user = User(
        username=user.username,
        data_url=user.data_url,
        role='client',
        hashed_password=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

# Login route
@user_router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    def is_invalid(db_user: User):
        not verify_password(user.password, db_user.hashed_password)
        
    return create_tokens(
        db,
        user.username,
        HTTPException(status_code=400, detail="Invalid credentials"),
        is_invalid
    )

@user_router.post("/refresh-token")
def refresh_token(token: RefreshToken, db: Session = Depends(get_db)):
    try:
        # Decode the refresh token
        payload = decode_refresh_token(token.refresh_token)
        username: str = payload.get("sub")
        
        expiry = datetime.fromtimestamp(payload.get('exp'))
        
        if(expiry > datetime.now()):
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            def validate_user(db_user: User):
                return db_user.refresh_token != token.refresh_token

            return create_tokens(
                db,
                username,
                HTTPException(status_code=401, detail="Invalid token"),
                validate_user
            )
        else:
            raise HTTPException(status_code=440, detail="Session Timed Out!")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


@user_router.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.refresh_token = None
    db.commit()
    return {"msg": "User logged out"}

# Protected API route
@user_router.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}