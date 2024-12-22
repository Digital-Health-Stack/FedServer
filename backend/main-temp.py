import asyncio
from datetime import datetime
import json
from typing import Dict
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import Null, and_, null, update
from sqlalchemy.orm import Session
# from models import User, Base
from schema import CreateFederatedLearning, ClientFederatedResponse, ClientModleIdResponse
from sse_starlette import EventSourceResponse
from utility.notification import add_notifications_for_recently_active_users
from models.FederatedSession import FederatedSession, FederatedSessionClient
from helpers.federated_learning import start_federated_learning
from helpers.websocket import ConnectionManager
from utility.FederatedLearning import FederatedLearning
from db import get_db
from models.User import User, Base
from schemas.UserSchema import RefreshToken, UserCreate, UserLogin
from helpers.auth import create_refresh_token, decode_refresh_token, get_password_hash, verify_password, create_access_token, get_current_user
from dotenv import load_dotenv
import jwt

from utility.user import get_unnotified_notifications
# from db import SessionLocal


load_dotenv()

# Create FastAPI app
app = FastAPI()

origins = [ 
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Or specify allowed methods, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Or specify allowed headers
)


# Create all tables
# Base.metadata.create_all(bind=engine)

websocket_manager = ConnectionManager()
        
# Signup route
@app.post("/signup", status_code=201)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # if db.query(User).filter(User.email == user.email).first():
    #     raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        username=user.username,
        data_url=user.data_url,
        hashed_password=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

# Login route
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    def is_invalid(db_user: User):
        not verify_password(user.password, db_user.hashed_password)
        
    return create_tokens(
        db,
        user.username,
        HTTPException(status_code=400, detail="Invalid credentials"),
        is_invalid
    )

@app.post("/refresh-token")
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

    
def create_tokens(db: Session, username: str, exception: HTTPException, check = null):
    # Verify if the refresh token exists in the database
    db_user = db.query(User).filter(User.username == username).first()
    if (not db_user) or (check and check(db_user)):
        raise exception
    
    # Create a new access token
    new_access_token = create_access_token(data={"sub": db_user.username})
    new_refresh_token = create_refresh_token(data={"sub": db_user.username})
    
    # Store refresh token in database
    db_user.refresh_token = new_refresh_token
    db.commit()

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@app.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.refresh_token = None
    db.commit()
    return {"msg": "User logged out"}

# Protected API route
@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}



@app.get("/notifications/stream")
async def notifications_stream(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    async def event_generator():
        while True:
            # Check if client has disconnected
            if await request.is_disconnected():
                break

            # Fetch notifications for the current user
            user_notifications = get_unnotified_notifications(user = current_user, db = db)
            
            if(len(user_notifications) > 0):
                data = [n.message for n in user_notifications]

                # Send the data as an SSE event
                yield {
                    "event": "new_notifications",   
                    "data": json.dumps(data),
                }
                

                for notification in user_notifications:
                    notification.notified_at = datetime.now()
                db.commit()

            # Wait for 5 seconds before sending the next update
            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())


federated_manager = FederatedLearning()

@app.post("/create-federated-session")
async def create_federated_session(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session: FederatedSession = federated_manager.create_federated_session(current_user, federated_details.fed_info, request.client.host)
    
    # await websocket_manager.broadcast({
    #     'type': "new-session",
    #     'message': "New Federated Session Avaliable!"
    # })
    
    message = {
        'type': "new-session",
        'message': "New Federated Session Avaliable!",
        'session_id': session.id
    }
    
    add_notifications_for_recently_active_users(db=db, message=message, valid_until=session.wait_till, excluded_users=[current_user])
    
    try:
        background_tasks.add_task(start_federated_learning, federated_manager, current_user, session)
        print("Background Task Added")
    except Exception as e:
        print(f"An error occurred while adding background process {Exception}")
        return {"message": "An error occurred while starting federated learning."}
    
    return {
        "message": "Federated Session has been created!",
        "session_id": session.id
    }


@app.get('/get-all-federated-sessions')
def get_all_federated_session(current_user: User = Depends(get_current_user)):
    return [
        {
            'id': id,
            'training_status': training_status,
            'name': federated_info.get('organisation_name')
        }
        for [id, training_status, federated_info]
        in federated_manager.get_my_sessions(current_user)
    ]

@app.get('/get-federated-session/{session_id}')
def get_federated_session(session_id: int, current_user: User = Depends(get_current_user)):
    try:
        federated_session_data = federated_manager.get_session(session_id)
        client = next((client for client in federated_session_data.clients if client.user_id == current_user.id), None)

        federated_response = {
            'federated_info': federated_session_data.federated_info,
            'training_status': federated_session_data.training_status,
            'client_status': client.status if client else 1
        }

        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post('/submit-client-federated-response')
def submit_client_federated_response(client_response: ClientFederatedResponse, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    '''
        decision : 1 means client accepts and 0 means rejects
        client_status = 2 means client has accepted the request
        client_status = 3 means client rejected the request
    '''
    session_id = client_response.session_id
    decision = client_response.decision
    client_status = 3
    if decision == 1:
        client_status = 2
    
    session = federated_manager.get_session(session_id)
    
    if(session):
        client = db.query(FederatedSessionClient).filter_by(session_id = session_id, user_id = current_user.id).first()
        if not client:
            federated_session_client = FederatedSessionClient(
                user_id = current_user.id,
                session_id = session_id,
                status = client_status,
                ip = request.client.host
            )
            
            db.add(federated_session_client)
        else:
            client.status = client_status
        db.commit()
    
    return { 'success': True, 'message': 'Client Decision has been saved'}


@app.post('/update-client-status-four')
def update_client_status_four(request: ClientModleIdResponse, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    '''
        Client have received the model parameters and waiting for server to start training
    '''

    session_id = request.session_id
    local_model_id = request.local_model_id
    
    db.execute(
        update(FederatedSessionClient)
        .where(and_(
            FederatedSessionClient.user_id == current_user.id,
            FederatedSessionClient.session_id == session_id
        ))
        .values(
            status = 4,
            local_model_id = local_model_id
        )
    )
    
    db.commit()

    return {'message': 'Client Status Updated to 4'}