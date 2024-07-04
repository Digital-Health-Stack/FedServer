from fastapi import FastAPI,BackgroundTasks,WebSocket,WebSocketDisconnect
from . import schema
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from .utility.Server import Server
import asyncio
import json
import uuid
import random

event = asyncio.Event()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user = []
round = 1
global_parameters = {}
max_round = 5
server = Server(global_parameters, max_round)
client_dict: {}
FedLearningInfo = {}


def generate_client_id():
    client_id = random.randint(1000, 99999)
    while client_dict[client_id] is not None:
        client_id = random.randint(1000, 99999)
    client_dict[client_id] = True
    return client_id


async def event_generator(data, msg):
    while True:
        await event.wait()
        event.clear()
        yield {
            "message": msg,
            "data": data
        }


# @app.get('/events')
# async def sse_endpoint():
#     return EventSourceResponse(event_generator())

@app.post('/sign-in')
def signIn(request: schema.User):
    user.append(request)
    print(f"{request.name} is registered")
    return {"message": "Client Registered Successfully", "ClientID": generate_client_id()}


@app.post('/request-federated-learning')
def requestFederatedLearning(request: schema.FederatedLearningInfo):
    FedLearningInfo = request
    # EventSourceResponse(event_generator("", "new_request_arrived"))
    return {"message": "Request Received Successfully"}


@app.get('/get-parameters')
def getParameters():
    obj = {
        "is_first": 1 if server.curr_round == 1 else 0,
        "parameter": server.globals_parameters,
        "round_num": server.curr_round
    }
    return obj


@app.post('/receive-parameters')
def receiveParameters(request: schema.Parameter):
    server.client_parameters[request.client_id] = request.client_parameter
    return {'message': 'Parameters are received!'}


def receive_parameters_client(server):
    event_generator("", f"ask_parameters_round_{server.curr_round}")
    while len(server.client_parameters) != len(user):
        pass


@app.get('/federated-learning')
async def federatedLearning(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_federated_learning)
    return {"status": "Federated Learning started"}


async def start_federated_learning():
    for i in range(1, server.max_round):
        print("-" * 50)
        server.curr_round = i
        print(f"Round {i}")
        print("-" * 50)
        receive_parameters_client(server)
        # Aggregate
        server.aggregate_weights_fedAvg_Neural()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.clients = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        # Generate unique id for client
        self.active_connections.append(websocket)
        self.clients[client_id] = websocket
        return client_id
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self,message: str,client_id: str):
        websocket = self.clients.get(client_id)
        if websocket:
            await websocket.send_text(message)
    
    async def findInterestedClients(self,parameters,client_id):
        new_message = {
            'parameters':parameters,
            'message':'Are you interested in doing Federated Learning'
        }
        message_str = json.dumps(new_message)
        for id,connection in self.clients.items():
            if client_id != id:
                await connection.send_text(message_str)
        
    async def broadcast(self,message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/master-ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    message = json.dumps({"message": f"Connected with client ID: {client_id} to Master-ws"})
    await websocket.send_text(message)
    try:
        while True:
            data = await websocket.receive_text()
            # await manager.broadcast(f"Message from {client_id}: {data}")
            await manager.findInterestedClients(f"Parameters from {client_id}:{data}",client_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")


        


