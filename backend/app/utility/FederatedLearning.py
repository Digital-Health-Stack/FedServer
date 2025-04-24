from datetime import datetime
from operator import or_
from typing import Dict, List,  Optional, Literal
from schema import FederatedLearningInfo, User
from sqlalchemy import and_, desc, select
from models.FederatedSession import FederatedSession, FederatedSessionClient, GlobalModelWeights, FederatedRoundClientSubmission, FederatedSessionLog
import numpy as np
from models import User as UserModel
from sqlalchemy.orm import Session, joinedload
from utility.db import engine
import json
from fastapi import HTTPException
import multiprocessing
from typing import Dict
import time


class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}
        self.processes: Dict[int, multiprocessing.Process] = {}
        self.process_start_times: Dict[int, float] = {}
        
    def add_process(self, session_id: int, process: multiprocessing.Process):
        """
        Add a process to the process manager for a specific session.
        
        Args:
            session_id (int): The ID of the federated session
            process (multiprocessing.Process): The process to manage
        """
        self.processes[session_id] = process
        self.process_start_times[session_id] = time.time()
        self.log_event(session_id, f"Added process {process.pid} for session")
        
    
    def get_process(self, session_id: int) -> multiprocessing.Process:
        """
        Get the process for a specific session.
        
        Args:
            session_id (int): The ID of the federated session
            
        Returns:
            multiprocessing.Process: The process for the session
        """
        return self.processes.get(session_id)
    
    def get_process_status(self, session_id: int) -> Dict:
        """
        Get status of a federated learning process using only stdlib
        
        Returns:
            Dict: {
                "exists": bool,
                "alive": bool,
                "pid": Optional[int],
                "status": Literal["running", "stopped", "unknown"],
                "start_time": Optional[str],
                "exit_code": Optional[int],
                "duration_seconds": Optional[float]
            }
        """
        process = self.get_process(session_id)
        if not process:
            return {"exists": False}
        
        is_alive = process.is_alive()
        status_info = {
            "exists": True,
            "pid": process.pid,
            "alive": is_alive,
            "status": "running" if is_alive else "stopped",
            "start_time": datetime.fromtimestamp(
                self.process_start_times.get(session_id, time.time())
            ).isoformat(),
            "exit_code": None if is_alive else process.exitcode,
            "duration_seconds": round(time.time() - self.process_start_times.get(session_id, time.time()), 2)
        }
        return status_info
    
    def get_combined_session_status(self, session_id: int, db: Session) -> Dict:
        """
        Get complete status including both session and process info
        
        Args:
            session_id: ID of the federated session
            db: Database session
            
        Returns:
            Dict: Combined status information
        """
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get basic session info
        session_status = {
            "session_id": session_id,
            "training_status": session.training_status,
            "current_round": session.curr_round,
            "max_rounds": session.max_round,
            "created_at": session.createdAt.isoformat(),
            "updated_at": session.updatedAt.isoformat() if session.updatedAt else None
        }
        
        # Add process info
        session_status["process"] = self.get_process_status(session_id)
        
        # Add recent logs (last 5)
        logs = db.query(FederatedSessionLog)\
                .filter(FederatedSessionLog.session_id == session_id)\
                .order_by(FederatedSessionLog.createdAt.desc())\
                .limit(5)\
                .all()
        
        session_status["recent_logs"] = [log.message for log in logs]
        
        return session_status
    
    
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


