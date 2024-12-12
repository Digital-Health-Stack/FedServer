import asyncio
from datetime import datetime
import json
from typing import Dict
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import Null, null
from sqlalchemy.orm import Session
# from models import User, Base
from schema import CreateFederatedLearning, ClientFederatedResponse, ClientModleIdResponse
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


# @app.websocket("/ws/notifications")
# async def websocket_endpoint(websocket: WebSocket, token = str):
#     """
#     WebSocket endpoint where each user connects with their unique ID.
#     """
#     # Verify the token and user_id here (for example, using JWT)
#     # For simplicity, we assume the token matches the user_id in this example.
    
#     current_user = await get_current_user(token)
    
#     if current_user:
#         await websocket_manager.connect(websocket, current_user.id)
#         try:
#             while True:
#                 # The connection remains open, waiting for notifications
#                 await websocket.receive_text()  # This can be used if clients send requests or status updates
#         except WebSocketDisconnect:
#             websocket_manager.disconnect(current_user.id)
            
#     raise HTTPException(status_code=401, detail="Unauthorized")

user_queues: Dict[str, asyncio.Queue] = {}

def get_or_create_user_queue(user_queues: Dict[str, asyncio.Queue], user_id: str):
    if user_id not in user_queues:
        user_queues[user_id] = asyncio.Queue()
    return user_queues[user_id]

async def event_stream(user_queues: Dict[str, asyncio.Queue], user: User):
    queue = get_or_create_user_queue(user_queues, user.id)
    try:
        while True:
            # Generate data for the SSE
            event = await queue.get()
            yield f"data: {event}\n\n"
    except asyncio.CancelledError:
        # Cleanup on disconnect
        user_queues.pop(user.id, None)

# Server Sent Events
@app.get("/notifications")
async def sse_notifications(token: str):
    current_user = await get_current_user(token)
    print(f"user id {current_user.id}")

    if current_user:
        return StreamingResponse(event_stream(user_queues, current_user), media_type="text/event-stream")

    raise HTTPException(status_code=401, detail="Unauthorized")


federated_manager = FederatedLearning()

@app.post("/create-federated-session")
async def create_federated_session(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
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
    
    for client_id in user_queues:
        await user_queues[client_id].put(json.dumps(message))
    
    try:
        background_tasks.add_task(start_federated_learning, federated_manager, current_user, session, user_queues)
        print("Background Task Added")
    except Exception as e:
        print(f"An error occurred while adding background process {Exception}")
        return {"message": "An error occurred while starting federated learning."}
    
    return {
        "message": "Federated Session has been created!",
        "session_id": session.id
    }

    # add_interested_user_to_session(federated_details.client_token, session_token, request, admin=True)
    # try:
    #     background_tasks.add_task(start_federated_learning, session_token)
    #     print("Background Task Added")
    # except Exception as e:
    #     print(f"An error occurred while adding background process {Exception}")
    #     return {"message": "An error occurred while starting federated learning."}


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
        client = next((client for client in federated_session_data.clients if client.client_id == current_user.id), None)

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
        # federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 2
        # add_interested_user_to_session(client_id, session_id, request, admin=False)
    # else:
    #     federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 3
    
    session = federated_manager.get_session(session_id)
    
    if(session):
        client = db.query(FederatedSessionClient).filter_by(session_id = session_id, client_id = current_user.id).first()
        if not client:
            federated_session_client = FederatedSessionClient(
                client_id = current_user.id,
                session_id = session_id,
                status = client_status,
                ip = request.client.host
            )
            
            db.add(federated_session_client)
        else:
            client.status = client_status
        db.commit()
    
    return { 'success': True, 'message': 'Client Decision has been saved'}

@app.post('/save-local-model-id')
def update_client_status_four(request: ClientModleIdResponse, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    '''
        Client have received the model parameters and waiting for server to start training
    '''
    local_model_id = request.local_model_id
    session_id = request.session_id
    
    client = db.query(FederatedSessionClient).filter_by(session_id = session_id, client_id = current_user.id, local_model_id = Null).first()
    
    if(client):
        client.local_model_id = local_model_id
        db.commit()

    # federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 4
    return {'message': 'Local model ID saved.'}