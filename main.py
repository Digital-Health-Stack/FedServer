from fastapi import FastAPI, BackgroundTasks
from . import schema
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from .utility.Server import Server
import asyncio
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


@app.get('/events')
async def sse_endpoint():
    return EventSourceResponse(event_generator("", "EventSource connected"))


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
