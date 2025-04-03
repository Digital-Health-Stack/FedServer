from fastapi import APIRouter, Depends, Request, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, update
from utility.db import get_db
from utility.FederatedLearning import FederatedLearning
from utility.auth import role
from utility.federated_learning import start_federated_learning
import json
import os

from models.FederatedSession import FederatedSession, FederatedSessionClient
from models.User import User

from schemas.user import ClientSessionStatusSchema
from schema import CreateFederatedLearning, ClientFederatedResponse, ClientModleIdResponse, ClientReceiveParameters

federated_router = APIRouter()
federated_manager = FederatedLearning()

@federated_router.get("/client/initiated_sessions", response_model=list[ClientSessionStatusSchema])
def get_initiated_jobs(current_user: User = Depends(role("client")), db: Session = Depends(get_db)):
    sessions = db.query(
            FederatedSession.id,
            FederatedSession.curr_round,
            FederatedSession.max_round,
            FederatedSession.session_price,
            FederatedSession.training_status,
            FederatedSessionClient.status.label('client_status')
        ).outerjoin(
            FederatedSessionClient,
            (FederatedSession.id == FederatedSessionClient.session_id) &
            (FederatedSessionClient.user_id == current_user.id)
        ).filter(FederatedSession.admin_id == current_user.id).all()
    
    return [ClientSessionStatusSchema.model_validate(dict(zip(
            ["session_id","curr_round", "max_round", "session_price", "training_status", "client_status"], session
        ))) for session in sessions]

@federated_router.get("/client/participated_sessions", response_model = list[ClientSessionStatusSchema])
def get_participated_sessions(current_user: User = Depends(role("client")), db: Session = Depends(get_db)):
    sessions = db.query(
        FederatedSession.curr_round,
        FederatedSession.max_round,
        FederatedSession.session_price,
        FederatedSession.training_status,
        FederatedSessionClient.status.label("client_status")
    ).join(
        FederatedSessionClient, FederatedSession.id == FederatedSessionClient.session_id
    ).filter(
        FederatedSessionClient.user_id == current_user.id
    ).all()
    return [ClientSessionStatusSchema.model_validate(dict(zip(
            ["curr_round", "max_round", "session_price", "training_status", "client_status"], session
        ))) for session in sessions]


@federated_router.post("/create-federated-session")
async def create_federated_session(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client"))
):  
    # Remove empty layers
    federated_details.fed_info.model_info["layers"] = [
        layer for layer in federated_details.fed_info.model_info["layers"] if layer.get("layer_type")
    ]
    session: FederatedSession = federated_manager.create_federated_session(current_user, federated_details.fed_info, request.client.host,db)
    
    # session.log_event(db, f"Federated session created by admin {current_user.id} from {request.client.host}")
    
    try:
        background_tasks.add_task(start_federated_learning, federated_manager, current_user, session, db)
        # session.log_event(db, "Background task for federated learning started")
    except Exception as e:
        # session.log_event(db, f"Error starting background task: {str(e)}")
        return {"message": "An error occurred while starting federated learning."}
    
    return {
        "message": "Federated Session has been created!",
        "session_id": session.id
    }


@federated_router.get('/get-all-federated-sessions')
def get_all_federated_session(current_user: User = Depends(role("client"))):
    return [
        {
            'id': id,
            'training_status': training_status,
            'name': federated_info.get('organisation_name')
        }
        for [id, training_status, federated_info]
        in federated_manager.get_my_sessions(current_user)
    ]

@federated_router.get('/get-federated-session/{session_id}')
def get_federated_session(session_id: int, current_user: User = Depends(role("client"))):
    try:
        federated_session_data = federated_manager.get_session(session_id)
        client = next((client for client in federated_session_data.clients if client.user_id == current_user.id), None)

        federated_response = {
            'federated_info': federated_session_data.federated_info,
            'training_status': federated_session_data.training_status,
            'client_status': client.status if client else 1,
            'session_price': federated_session_data.session_price
        }

        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    

##*************** Do i need client dependency here ?***************
@federated_router.post('/submit-client-price-response')
def submit_client_price_response(client_response: ClientFederatedResponse, request: Request, db: Session = Depends(get_db)):
    '''
        decision : 1 means client accepts the price, -1 means client rejects the price
        training_status = 2 means the training process should start
    '''
    try:
        session_id = client_response.session_id
        decision = client_response.decision
        
        session = federated_manager.get_session(session_id)
        if(session):
            # Fetch the FederatedSession by session_id
            federated_session = db.query(FederatedSession).filter_by(id = session_id).first()
            if not federated_session:
                raise HTTPException(status_code=404, detail="Federated session not found")
            # Update training_status based on the decision
            if decision == 1:
                federated_session.training_status = 2  # Update training_status to 2 (start training)
            elif decision == -1:
                federated_session.training_status = -1  # Keep or set to a default status for rejection
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid decision value. Must be 1 (accept) or -1 (reject)."
                )
            # Commit changes to the database
            db.commit()
            return {'success': True, 'message': 'Training status updated successfully'}
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@federated_router.post('/submit-client-federated-response')
def submit_client_federated_response(client_response: ClientFederatedResponse, request: Request, current_user: User = Depends(role("client")), db: Session = Depends(get_db)):
    '''
        decision : 1 means client accepts and 0 means rejects
        client_status = 2 means client has accepted the request
        client_status = 3 means client rejected the request
    '''
    session_id = client_response.session_id
    decision = client_response.decision
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


@federated_router.post('/update-client-status-four')
def update_client_status_four(request: ClientModleIdResponse, current_user: User = Depends(role("client")), db: Session = Depends(get_db)):
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

@federated_router.get('/get-model-parameters/{session_id}')
def get_model_parameters(session_id: str):
    '''
        Client have received the model parameters and waiting for server to start training
    '''
    global_parameters = json.loads(federated_manager.get_session(session_id).global_parameters)
    
    response_data = {
        "global_parameters": global_parameters,
        "is_first": 0
    }

    # Save global_parameters string into a file
    file_path = "global_parameters.txt"  # Specify the desired file path and name
    with open(file_path, "a") as file:
        file.write("\n---\n")  # Add a separator before each new entry
        file.write(json.dumps(global_parameters))  # Append the JSON string
        file.write("\n")  # Add a newline after the entry for readability
    print(f"Global parameters have been saved to {file_path}.")
    return response_data

@federated_router.post('/receive-client-parameters')
def receive_client_parameters(request: ClientReceiveParameters,  current_user: User = Depends(role("client")), db: Session = Depends(get_db)):
    session_id = request.session_id
    client_parameter = request.client_parameter
    
    session_data = db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Federated Session with ID {session_id} not found!")
    
    # Deserialize client_parameters from JSON to a Python dictionary
    existing_parameters = json.loads(session_data.client_parameters) if session_data.client_parameters else {}
    
    existing_parameters[str(current_user.id)] = client_parameter
    session_data.client_parameters = json.dumps(existing_parameters)
    
    db.commit()
    
    # federated_manager.federated_sessions[session_id]['client_parameters'][client_id] = request.client_parameter
    return {"message": "Client Parameters Received"}

@federated_router.get('/get-all-completed-trainings')
def get_training_results():
    # iterate ove Global_test_results folder and return the completed sessions' results
    try:
        results_dir = "Global_test_results"
        results = []
        for file in os.listdir(results_dir):
            if file.endswith(".json"):
                with open(os.path.join(results_dir, file), "r") as f:
                    result = json.load(f)
                    # return only session_id and organisation_name
                    # only save session_id from file name not all filename
                    results.append({
                        "session_id": file.split("_")[0],
                        "org_name": result["session_data"]["organisation_name"]
                    })
        return {"results": results}

    except Exception as e:
        return {"message": f"No training results"}


@federated_router.get('/get-training-result/{session_id}')
def get_training_results(session_id: str):
    # iterate ove Global_test_results folder and return the session_id's results
    try:
        results_dir = "Global_test_results"
        with open(os.path.join(results_dir, f"{session_id}_test_results.json"), "r") as f:
            result = json.load(f)
            return result
    except Exception as e:
        return {"message": f"No training results with this session_id"}