from fastapi import FastAPI,BackgroundTasks,Request,HTTPException
from .Schema import FederatedLearningInfo, User, Parameter, CreateFederatedLearning, ClientFederatedResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from .utility.Server import Server
import asyncio
import json
import uuid
import random
from typing import Dict,List
import asyncio
import numpy as np

'''
    Naming Conventions as per PEP- https://peps.python.org/pep-0008/#function-and-variable-names
    Classes - CapWords Convention
    Methods - Function names should be lowercase, with words separated by underscores as necessary to improve readability.
    Variables - Variable names follow the same convention as function names.
'''

event = asyncio.Event()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User array contains tokens for all users
clients_data = []
round = 1  #
global_parameters = {} #
max_round = 5 #
server = Server(global_parameters, max_round)
client_dict =  {}   #
FedLearningInfo = {}

# Utility Function
def generate_unique_token():
    token = str(uuid.uuid4())
    return token

def generate_session_token():
    session_token = generate_unique_token()
    # Check if token is already assigned to other federated session
    return session_token

def generate_user_token():
    user_token = generate_unique_token()
    # Check if token is already assigned to other federated session
    return user_token
    
class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}
    
    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(self,session_id: str,federated_info: FederatedLearningInfo) :
        """
        Creates a new federated learning session.

        Parameters:
        - session_id (str): Unique identifier for the session.
        - federated_info (FederatedLearningInfo): Information about the federated learning session.
        - clients_data (list): List of client user IDs participating in the session.

        Initializes session with default values:
        - admin: None (can be assigned later)
        - curr_round: 1 (current round number)
        - max_round: 5 (maximum number of rounds)
        - interested_clients: Empty dictionary to store IDs of interested clients
        - global_parameters: Empty list to store global parameters
        - clients_status: Dictionary to track status of all clients. 
                          Status values: 1 (not responded), 2 (accepted), 3 (rejected)
        - training_status: 1 (server waiting for all clients), 2 (training starts)
        """
        self.federated_sessions[session_id] = {
            "federated_info": federated_info,
            # "active_clients": [],
            # "client_sockets": {},
            "admin": None,
            "curr_round": 1,
            "max_round": 5,
            "interested_clients": {}, # contains ids of interested_clients
            "global_parameters": [],    # Contains user id of interested students
            "clients_status": {user_id: {"status": 1} for user_id in clients_data},   
            "training_status": 1,            # 1 for server waiting for all clients and 2 for training starts
            "client_parameters": {}
        }
    
    def get_session(self, session_id: str) -> FederatedLearningInfo:
        """
        Retrieves information about a federated learning session.

        Parameters:
        - session_id (str): Unique identifier for the session.

        Returns:
        - FederatedLearningInfo: Information about the federated learning session.
        """
        return self.federated_sessions[session_id]["federated_info"]
    
    def clear_client_parameters(self,session_id: str):
        self.federated_sessions[session_id]['client_parameters'] = {}

    def aggregate_weights_fedAvg_Neural(self,session_id:str):
        # Initialize a dictionary to hold the aggregated sums of vectors
        # print("Received Parameters : " , type(self.client_parameters[1][1][0]),
        #                                 len(self.client_parameters[1][1]),self.client_parameters[1][2][0][:5])

        # Count the number of clients
        num_interested_clients = len(self.federated_sessions[session_id]['client_parameters'])

        client_parameters = self.federated_sessions[session_id]['client_parameters']
        for client in client_parameters:
            client_parameters[client] = [np.array(arr) for arr in client_parameters[client]]

        aggregated_sums = []
        for layer in range(len(client_parameters[1])):
            layer_dimension = client_parameters[1][layer].shape

            aggregated_layer = np.zeros(layer_dimension)

            for client in client_parameters:
                aggregated_layer += client_parameters[client][layer]
            aggregated_layer /= num_interested_clients
            aggregated_sums.append(aggregated_layer.tolist())

        print("Aggregate Weights after FedAvg: ",type(aggregated_sums[0][1][0]),len(aggregated_sums[0][1]),aggregated_sums[2][0][:4])

        self.federated_sessions[session_id]['globals_parameters'] = aggregated_sums
        self.clear_client_parameters(session_id)
    
    

# Usage
federated_manager = FederatedLearning()

async def wait_for_client_confirmation(session_id: str):
    session_data = federated_manager.federated_sessions[session_id]
    all_ready_for_training = False

    while not all_ready_for_training:
        await asyncio.sleep(5)
        all_ready_for_training = all(client['status']!=1 for client in session_data['client_status'].values())
        print("Waiting for client confirmations....Stage 1")

async def wait_for_clients_training(session_id: str):
    session_data = federated_manager.federated_sessions[session_id]
    num_interested_clients = len(session_data['interested_clients'])

    while len(session_data['client_parameters']) < num_interested_clients:
        await asyncio.sleep(5)
        print(f"Waiting for {num_interested_clients - len(session_data['client_parameters'])}")
    
    print("All clients have sent parameters. Starting Aggregation...")



async def start_federated_learning(session_id: str):
    """
    Background task to manage federated learning rounds.

    This function runs in the background, waiting for client responses before proceeding with each round
    of federated learning.

    Each round consists of:
    1. Setting the current round number (`curr_round`) in the server.
    2. Printing round information.
    3. Receiving parameters from clients.
    4. Aggregating weights using federated averaging with neural networks.

    """
    session_data = federated_manager.federated_sessions[session_id] # session_data points to same object variables in Python that point to mutable objects (like dictionaries) actually reference the same underlying object in memory.
    await wait_for_client_confirmation(session_id)

    # Send a signal to client using websocket so that client 
    # can get parameters of the model and he can start a background response
    # Do a handshake whether client has model to train and he started a background 
    # process using a websocket
    for i in range(0, session_data['max_round']):
        print("-" * 50)
        server.curr_round = i
        print(f"Round {i+1}")
        print("-" * 50)
        
        await wait_for_clients_training(session_id)
        # Aggregate
        federated_manager.aggregate_weights_fedAvg_Neural(session_id)


def add_interested_user_to_session(client_token,session_token: str, request: Request,admin):
        """
        Generates a token for user which will be used to validate the sender, and the token will be bound to a spefic session.
        :param name: This will be authentication token for a user in future when establish authentication each user has a unique user_id
        :param admin: This user is admin of this request or not
        :param request: 
        :return:
        """ 
        federated_manager.federated_sessions[session_token]['clients_status'][client_token]['status'] = 2
        federated_manager.federated_sessions[session_token]["interested_clients"][client_token] = {
            "ip": request.client.host,
        }
        if admin:
            federated_manager.federated_sessions[session_token]["admin"] = client_token


@app.post('/sign-in')
def signIn(request: User):
    user_token = generate_user_token()
    clients_data.append(user_token)
    print(f"{request.name} is registered")
    return {"message": "Client Registered Successfully", "client_token": user_token}

@app.post("/create-federated-session")
def create_federated_session(federated_details: CreateFederatedLearning,request: Request,background_tasks: BackgroundTasks):
    session_token = generate_session_token()
    federated_manager.create_federated_session(session_token,federated_details.fed_info)
    
    add_interested_user_to_session(federated_details.client_token,session_token,request,admin=True)
    background_tasks.add_task(start_federated_learning)
    return {"message": "Federated Session has been created!"}

@app.get('/get-all-federated-sessions')
def get_all_federated_session():
    federated_session = []
    for session_id,session_data in federated_manager.federated_sessions.items():
        federated_session.append({
            'session_id' : session_id,
            "training_status" : session_data["training_status"]
        })
    return {"federated_session": federated_session}

@app.get('/get-federated-session/{session_id}')
def get_federated_session(session_id: str,client_id: str):
    first_session_key = list(federated_manager.federated_sessions.keys())[0]
    try: 
        federated_session_data = federated_manager.federated_sessions[session_id]
        federated_response = {
            'federated_info': federated_session_data['federated_info'],
            'training_status': federated_session_data['training_status'],
            'client_status': federated_session_data['clients_status'][client_id]['status']
        }
        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post('/submit-client-federated-response')
def submit_client_federated_response(client_response: ClientFederatedResponse,request: Request):
    '''
        decision : 1 means client accepts and 0 means rejects
    '''
    try:
        session_id = client_response.session_id
        client_id = client_response.client_id
        decision = client_response.decision
        if decision==1:

            federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 2
            add_interested_user_to_session(client_id,client_id,request,admin=False)
        else:
               federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 3
        return {'message': 'Client Decision has been saved'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



        


