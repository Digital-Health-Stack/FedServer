from datetime import datetime
from operator import or_
from typing import Dict, List, Optional, Literal
from schema import CreateFederatedLearning, FederatedLearningInfo, User
from sqlalchemy import and_, desc, select
from models.FederatedSession import (
    FederatedSession,
    FederatedSessionClient,
    FederatedRoundClientSubmission,
    FederatedSessionLog,
)
import numpy as np
from models import User as UserModel
from sqlalchemy.orm import Session, joinedload
from utility.db import engine
import json
from fastapi import HTTPException
import multiprocessing
from typing import Dict
import time
from pathlib import Path
import shutil
from constant.enums import FederatedSessionLogTag


class FederatedLearning:
    def __init__(self):
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
        self.log_event(
            session_id,
            f"Added process {process.pid} for session",
            FederatedSessionLogTag.INFO,
        )

    def get_process(self, session_id: int) -> Optional[multiprocessing.Process]:
        """
        Get the process for a specific session.

        Args:
            session_id (int): The ID of the federated session

        Returns:
            Optional[multiprocessing.Process]: The process for the session, or None if not found
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
            "duration_seconds": round(
                time.time() - self.process_start_times.get(session_id, time.time()), 2
            ),
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
            "updated_at": session.updatedAt.isoformat() if session.updatedAt else None,
        }

        # Add process info
        session_status["process"] = self.get_process_status(session_id)

        # Add recent logs (last 5)
        logs = (
            db.query(FederatedSessionLog)
            .filter(FederatedSessionLog.session_id == session_id)
            .order_by(FederatedSessionLog.createdAt.desc())
            .limit(5)
            .all()
        )

        session_status["recent_logs"] = [log.message for log in logs]

        return session_status

    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(
        self, user: UserModel, federated_info: CreateFederatedLearning, ip, db: Session
    ):
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
            admin_id=user.id,
        )
        try:
            db.add(federated_session)
            db.commit()
            db.refresh(federated_session)
        except Exception as e:
            db.rollback()  # Rollback in case of any error
            raise HTTPException(
                status_code=500, detail=f"Failed to create federated session: {str(e)}"
            )

        federated_session_client = FederatedSessionClient(
            user_id=user.id, session_id=federated_session.id, status=0, ip=ip
        )
        db.add(federated_session_client)
        db.commit()
        return federated_session

    def get_session(self, federated_session_id: int) -> FederatedSession:
        """
        Retrieves information about a federated learning session.

        Parameters:
        - session_id (str): Unique identifier for the session.

        Returns:
        - FederatedLearningInfo: Information about the federated learning session.
        """

        with Session(engine) as db:
            stmt = (
                select(FederatedSession, FederatedSession.clients)
                .where(FederatedSession.id == federated_session_id)
                .options(joinedload(FederatedSession.clients))
            )
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
                FederatedSession.createdAt,
            ).order_by(desc(FederatedSession.createdAt))

            federated_sessions = db.execute(stmt).all()
            return federated_sessions

    def get_my_sessions(self, user: UserModel):
        with Session(engine) as db:
            stmt = (
                select(
                    FederatedSession.id,
                    FederatedSession.training_status,
                    FederatedSession.federated_info,
                )
                .join(FederatedSession.clients)
                .order_by(desc(FederatedSession.createdAt))
                .where(
                    or_(
                        FederatedSession.federated_info.wait_time > datetime.now(),
                        FederatedSessionClient.user_id == user.id,
                    )
                )
            )

            federated_sessions = db.execute(stmt).all()

            return federated_sessions

    def clear_client_parameters(self, session_id: str, round_number: int):
        """
        Clears client parameters for a specific session and round
        by deleting related FederatedRoundClientSubmission and ClientModelWeight entries.
        """
        # First clear filesystem parameters
        local_dir = Path(f"tmp/parameters/{session_id}/local")
        if local_dir.exists():
            try:
                shutil.rmtree(local_dir)
                print(
                    f"Successfully deleted all local parameters for session {session_id}"
                )
                return True
            except Exception as e:
                print(f"Failed to delete {local_dir}: {str(e)}")
                return False
        else:
            print(f"Local directory {local_dir} doesn't exist - nothing to delete")
            return False

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

        # Define paths
        base_dir = Path(f"tmp/parameters/{session_id}")
        local_dir = base_dir / "local"
        global_dir = base_dir / "global"
        global_weights_file = global_dir / "global_weights.json"
        # Create global directory if it doesn't exist
        global_dir.mkdir(parents=True, exist_ok=True)

        # Get all client parameter files for this round
        client_files = list(local_dir.glob("*.json"))
        client_files = [
            f for f in client_files if not f.name.endswith("_metadata.json")
        ]

        if not client_files:
            print("No client submissions found for this round.")
            return

        # Initialize aggregated sums
        aggregated_sums = None
        num_clients = 0

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
                return [
                    average_parameters(sub_aggregated, count)
                    for sub_aggregated in aggregated
                ]
            else:
                return (aggregated / count).tolist()

        # for submission in submissions:
        #     weights = submission.model_weights.weights
        #     if not aggregated_sums:
        #         for key in weights:
        #             aggregated_sums[key] = initialize_aggregated_sums(weights[key])

        #     for key in weights:
        #         aggregated_sums[key] = sum_parameters(aggregated_sums[key], weights[key])

        # Process each client's parameters
        for client_file in client_files:
            # Read client parameters
            with open(client_file, "r") as f:
                client_weights = json.load(f)

            # Read metadata to verify round number
            metadata_file = local_dir / f"{client_file.stem}_metadata.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if metadata.get("round_number") != round_number:
                continue  # Skip if not for current round

            num_clients += 1

            # Initialize aggregated_sums structure on first client
            if aggregated_sums is None:
                aggregated_sums = {}
                for key in client_weights:
                    aggregated_sums[key] = initialize_aggregated_sums(
                        client_weights[key]
                    )

            # Sum the parameters
            for key in client_weights:
                aggregated_sums[key] = sum_parameters(
                    aggregated_sums[key], client_weights[key]
                )

        if num_clients == 0:
            print("No valid client submissions found for this round.")
            return

        if aggregated_sums is None:
            print("No aggregated sums available for averaging.")
            return

        # Average the parameters
        for key in aggregated_sums:
            aggregated_sums[key] = average_parameters(aggregated_sums[key], num_clients)

        # Save global weights to file
        with open(global_weights_file, "w") as f:
            json.dump(aggregated_sums, f)

        # Optionally, save metadata about the aggregation
        aggregation_metadata = {
            "session_id": session_id,
            "round_number": round_number,
            "num_clients": num_clients,
            "aggregated_at": datetime.utcnow().isoformat(),
        }

        with open(global_dir / "aggregation_metadata.json", "w") as f:
            json.dump(aggregation_metadata, f)
        return

    def log_event(self, session_id: int, message: str, tag: FederatedSessionLogTag):
        with Session(engine) as db:
            log_entry = FederatedSessionLog(
                session_id=session_id, message=message, tag=tag
            )
            db.add(log_entry)
            db.commit()
            print(f"[LOG] Session {session_id}: {message} ")

    def get_latest_global_weights(self, session_id: int):
        """
        Retrieve the latest global model weights for a given federated session.

        Args:
            session_id (int): The ID of the federated session.
            db (Session): SQLAlchemy database session.

        Returns:
            dict or None: The latest global weights as a dictionary, or None if not found.
        """
        global_weights_path = Path(
            f"tmp/parameters/{session_id}/global/global_weights.json"
        )
        if not global_weights_path.exists():
            print("File Federated Learning: Global Weights not found!!")
            return None
        try:
            with open(global_weights_path, "r") as f:
                weights = json.load(f)
            return weights
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading global weights: {str(e)}")
            return None
