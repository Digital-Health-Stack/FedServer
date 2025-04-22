from datetime import datetime
from operator import or_
from typing import Dict, List
from schema import FederatedLearningInfo, User
from sqlalchemy import and_, desc, select
from models.FederatedSession import FederatedSession, FederatedSessionClient, GlobalModelWeights, FederatedRoundClientSubmission, FederatedSessionLog
import numpy as np
from models import User as UserModel
from sqlalchemy.orm import Session, joinedload
from utility.db import engine
import json
from fastapi import HTTPException


class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}
    
    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(self, user: UserModel, federated_info: FederatedLearningInfo, ip, db:Session) :
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
        
        federated_session = FederatedSession(
            federated_info=federated_info.__dict__,
            admin_id = user.id,
        )
        try:
            db.add(federated_session)
            db.commit()
            db.refresh(federated_session)
        except Exception as e:
            db.rollback()  # Rollback in case of any error
            raise HTTPException(status_code=500, detail=f"Failed to create federated session: {str(e)}")    
            
        federated_session_client = FederatedSessionClient(
            user_id = user.id,
            session_id = federated_session.id,
            status = 0,
            ip = ip
        )
        db.add(federated_session_client)
        db.commit()
        # Create the global model weight for the session
        global_model_weight = GlobalModelWeights(
            session_id=federated_session.id,
            weights={},  # You can initialize this as an empty dictionary or with some default weights
        )
        db.add(global_model_weight)
        db.commit()
        return federated_session

    
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
            stmt = select(
                FederatedSession.id,
                FederatedSession.training_status,
                FederatedSession.federated_info,
                FederatedSession.createdAt
            ).order_by(desc(FederatedSession.createdAt))
        
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
    
    def clear_client_parameters(self, session_id: str, round_number: int):
        """
        Clears client parameters for a specific session and round
        by deleting related FederatedRoundClientSubmission and ClientModelWeight entries.
        """
        with Session(engine) as db:
            # Fetch all submissions for the current round
            submissions = db.query(FederatedRoundClientSubmission).filter_by(
                session_id=session_id,
                round_number=round_number
            ).all()

            if not submissions:
                print(f"No client submissions found for session {session_id} and round {round_number}. Nothing to clear.")
                return

            # Delete all fetched submissions (weights will cascade delete)
            for submission in submissions:
                db.delete(submission)

            db.commit()
            print(f"Cleared all client parameters for session {session_id} and round {round_number}.")

    def aggregate_weights_fedAvg_Neural(self, session_id: str, round_number: int):
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
            # Fetch all client submissions for this round
            submissions = db.query(FederatedRoundClientSubmission).filter_by(
                session_id=session_id,
                round_number=round_number
            ).all()
            if not submissions:
                raise ValueError("No client submissions found for this round.")
            
            # Initialize a dictionary to hold the aggregated sums of parameters
            num_clients = len(submissions)
            aggregated_sums = {}

            def initialize_aggregated_sums(param):
                if isinstance(param, list):
                    return [initialize_aggregated_sums(p) for p in param]
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

            for submission in submissions:
                weights = submission.model_weights.weights
                if not aggregated_sums:
                    for key in weights:
                        aggregated_sums[key] = initialize_aggregated_sums(weights[key])

                for key in weights:
                    aggregated_sums[key] = sum_parameters(aggregated_sums[key], weights[key])

            # Average the aggregated sums
            for key in aggregated_sums:
                aggregated_sums[key] = average_parameters(aggregated_sums[key], num_clients)


            # Replace the old GlobalModelWeights if it exists
            global_weight = db.query(GlobalModelWeights).filter_by(session_id=session_id).first()
            if global_weight:
                global_weight.weights = aggregated_sums
                global_weight.updated_at = datetime.now()  # Add this field if not present already
            else:
                global_weight = GlobalModelWeights(
                    session_id=session_id,
                    weights=aggregated_sums
                )
                db.add(global_weight)
                print(f"Created new global weight for session {session_id}")
            db.commit()

            # # Save aggregated_sums dictionary into a text file with appending
            # file_path = "aggregated_sums.txt"  # Specify the desired file path and name

            # # Convert the dictionary to a JSON string
            # aggregated_sums_str = json.dumps(aggregated_sums, indent=4)  # Format with indent for better readability

            # # Append aggregated_sums to the file with a separator       
            # with open(file_path, "a") as file:  # Use "a" mode to append
            #     file.write("\n---\n")  # Add a separator before each new entry
            #     file.write(aggregated_sums_str)  # Append the formatted JSON string
            #     file.write("\n")  # Add a newline after the entry for readability

            # print(f"Aggregated sums have been appended to {file_path} with a separator.")

    def log_event(self, session_id: int,message: str):
        with Session(engine) as db:
            log_entry = FederatedSessionLog(session_id=session_id, message=message)
            db.add(log_entry)
            db.commit()
            print(f"[LOG] Session {session_id}: {message} ")
    
    def get_latest_global_weights(self,session_id: int):
        """
        Retrieve the latest global model weights for a given federated session.

        Args:
            session_id (int): The ID of the federated session.
            db (Session): SQLAlchemy database session.

        Returns:
            dict or None: The latest global weights as a dictionary, or None if not found.
        """
        # Retrieve client parameters
        with Session(engine) as db:
            weights_entry = (
                db.query(GlobalModelWeights)
                .filter(GlobalModelWeights.session_id == session_id)
                .order_by(GlobalModelWeights.createdAt.desc())
                .first()
            )

            if weights_entry and weights_entry.weights:
                return weights_entry.weights
        return None


