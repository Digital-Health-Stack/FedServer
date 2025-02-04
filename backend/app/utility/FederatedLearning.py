from datetime import datetime
from operator import or_
from typing import Dict, List
from schema import FederatedLearningInfo, User
from sqlalchemy import and_, desc, select
from models.FederatedSession import FederatedSession, FederatedSessionClient
from utility.Server import Server
import numpy as np
from models import User as UserModel
from sqlalchemy.orm import Session, joinedload
from db import engine
import json


class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}
    
    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(self, user: UserModel, federated_info: FederatedLearningInfo, ip) :
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
        with Session(engine) as db:
            federated_session = FederatedSession(
                federated_info=federated_info.__dict__,
                admin_id = user.id,
            )
            
            db.add(federated_session)
            db.commit()
            db.refresh(federated_session)
            
            federated_session_client = FederatedSessionClient(
                user_id = user.id,
                session_id = federated_session.id,
                status = 2,
                ip = ip
            )
            
            db.add(federated_session_client)
            db.commit()

            federated_session.id

            return federated_session

        # self.federated_sessions[session_id] = {
        #     "federated_info": federated_info,
        #     "admin": None,
        #     "curr_round": 1,
        #     "max_round": 10,
        #     # "interested_clients": {}, # contains ids of interested_clients
        #     "global_parameters": [],   # contains global parameters
        #     "clients_status": {user_id: {"status": 1} for user_id in clients_data}, 
        #     "training_status": 1,            # 1 for server waiting for all clients and 2 for training starts
        #     "client_parameters": {}
        # }
    
    def get_session(self, federated_session_id: int) -> FederatedLearningInfo:
        """
        Retrieves information about a federated learning session.

        Parameters:
        - session_id (str): Unique identifier for the session.

        Returns:
        - FederatedLearningInfo: Information about the federated learning session.
        """
        # return self.federated_sessions[session_id]["federated_info"]
        
        with Session(engine) as db:
            stmt = select(FederatedSession, FederatedSession.clients).where(FederatedSession.id == federated_session_id).options(joinedload(FederatedSession.clients))
            federated_session = db.execute(stmt).scalar()
            
            return federated_session
        
    # def get_session_clients(self, federated_session_id: int):
    #     with Session(engine) as db:
    #         stmt = select(FederatedSessionClient).where(FederatedSession.id == )
            
    def get_all(self):
        with Session(engine) as db:
            stmt = select(FederatedSession.id, FederatedSession.training_status).order_by(desc(FederatedSession.createdAt))
            federated_sessions = db.execute(stmt).all()
            
            return federated_sessions
    
    def get_my_sessions(self, user: UserModel):
        with Session(engine) as db:
            stmt = select(
                FederatedSession.id,
                FederatedSession.training_status,
                FederatedSession.federated_info
            ).join(
                FederatedSession.clients
            ).order_by(
                desc(FederatedSession.createdAt)
            ).where(
                or_(
                    FederatedSession.wait_till > datetime.now(),
                    FederatedSessionClient.user_id == user.id
                )
            )

            federated_sessions = db.execute(stmt).all()
            
            return federated_sessions
    
    def clear_client_parameters(self,session_id: str):
        self.federated_sessions[session_id]['client_parameters'] = {}

    def aggregate_weights_fedAvg_Neural(self, session_id: str):
        """
        # ========================================================================================================
        # Expected Params config for each client to work Federated Averaging correctly
        # ========================================================================================================
        # Parameters expected for each client_parameter in the form of a dictionary:
        # for eg. 1 client parameter is like below, one or more than one keys can be there (based on model needs)
        # {
        #     "weights": [list of numpy arrays],
        #     "biases": [list of numpy arrays],
        #     "other_parameters": [list of numpy arrays]
        # }
        # ========================================================================================================
        """
        # Retrieve client parameters
        with Session(engine) as db:
            federated_session = db.query(FederatedSession).filter_by(id=session_id).first()
            
            if not federated_session:
                raise ValueError(f"FederatedSession with ID {session_id} not found.")
            # Count the number of clients
            num_interested_clients = len(federated_session.clients)
            client_parameters = json.loads(federated_session.client_parameters or "{}")  # Ensure client_parameters is a dict

            # Initialize a dictionary to hold the aggregated sums of parameters
            aggregated_sums = {}

            def initialize_aggregated_sums(param, aggregated):
                if isinstance(param, list):
                    return [initialize_aggregated_sums(sub_param, aggregated) for sub_param in param]
                else:
                    return np.zeros_like(param)

            def sum_parameters(aggregated, param):
                if isinstance(param, list):
                    for i in range(len(param)):
                        aggregated[i] = sum_parameters(aggregated[i], param[i])
                    return aggregated
                else:
                    return aggregated + np.array(param)

            def average_parameters(aggregated, count):
                if isinstance(aggregated, list):
                    return [average_parameters(sub_aggregated, count) for sub_aggregated in aggregated]
                else:
                    return (aggregated / count).tolist()

            # Iterate over each client's parameters
            for client in client_parameters:
                client_param = client_parameters[client]

                # Initialize aggregated_sums with the same structure as the first client's parameters
                if not aggregated_sums:
                    for key in client_param:
                        aggregated_sums[key] = initialize_aggregated_sums(client_param[key], aggregated_sums)

                # Sum the parameters for each key
                for key in client_param:
                    aggregated_sums[key] = sum_parameters(aggregated_sums[key], client_param[key])

            # Average the aggregated sums
            for key in aggregated_sums:
                aggregated_sums[key] = average_parameters(aggregated_sums[key], num_interested_clients)

            print("Aggregated Parameters after FedAvg:",
                  {k: (type(v), len(v) if isinstance(v, list) else 'N/A') for k, v in aggregated_sums.items()})

            # Save the aggregated parameters back to the session
            federated_session.global_parameters = json.dumps(aggregated_sums)

            # Save aggregated_sums dictionary into a text file with appending
            file_path = "aggregated_sums.txt"  # Specify the desired file path and name

            # Convert the dictionary to a JSON string
            aggregated_sums_str = json.dumps(aggregated_sums, indent=4)  # Format with indent for better readability

            # Append aggregated_sums to the file with a separator       
            with open(file_path, "a") as file:  # Use "a" mode to append
                file.write("\n---\n")  # Add a separator before each new entry
                file.write(aggregated_sums_str)  # Append the formatted JSON string
                file.write("\n")  # Add a newline after the entry for readability

            print(f"Aggregated sums have been appended to {file_path} with a separator.")
            db.commit()


