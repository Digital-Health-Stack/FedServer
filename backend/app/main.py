import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.dataset import dataset_router
from api.federated import federated_router
from api.notification import notifications_router
from api.user import user_router
from api.temporary import temporary_router
from api.confidential import confidential_router

load_dotenv()

# Create FastAPI app
app = FastAPI()

client1_url = os.getenv("CLIENT1_URL", "http://default-url.com")
origins = [ 
    "http://localhost:5173",
    "http://localhost:5174",
    client1_url  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Or specify allowed methods, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Or specify allowed headers
)

app.include_router(dataset_router,tags=["Dataset"])
app.include_router(user_router,tags=["User"])
app.include_router(federated_router,tags=["Federated"])
app.include_router(notifications_router,tags=["Notification"])
app.include_router(temporary_router,tags=["Temporary"])
app.include_router(confidential_router,tags=["Confidential"])