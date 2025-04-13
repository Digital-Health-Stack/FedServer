from fastapi import APIRouter, HTTPException
import os
from utility.auth import role
from fastapi import Depends
from models.User import User
import subprocess
import sys
from utility.auth import verify_token

temporary_router = APIRouter()

@temporary_router.get('/check')
def check():
    return {"message": "Everyone can access it!"}

@temporary_router.get('/check-client')
def check_client(client: User = Depends(role("client"))):
    return {"message": "Only clients can access it!"}

@temporary_router.get('/check-admin')
def check_admin(admin: User = Depends(role("admin"))):
    return {"message": "Only admins can access it!"}


@temporary_router.get("/check-current_user")
def check_user(token: str):
    current_user = verify_token(token)
    return current_user

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

def aggregate_weights_fedAvg_Neural( session_id: str, round_number: int):
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
            
            file_path = "logs/client_submissions_log.txt"

            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n--- Client Submissions Log ---\n")
                for submission in submissions:
                    user_id = submission.user_id  # or whatever field holds the user info
                    weights = submission.model_weights.weights
                    f.write(f"User ID: {user_id}\n")
                    f.write(f"Weights: {json.dumps(weights)}\n")
                    f.write("---\n")
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

            print("Aggregated Parameters after FedAvg:",
                  {k: (type(v), len(v) if isinstance(v, list) else 'N/A') for k, v in aggregated_sums.items()})
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n--- Aggregated Parameters after FedAvg ---\n")

                # Print type and length summary
                for k, v in aggregated_sums.items():
                    v_type = type(v)
                    v_len = len(v) if isinstance(v, list) else 'N/A'
                    f.write(f"{k}: Type: {v_type}, Length: {v_len}\n")

                # Print actual aggregated parameter values
                f.write("\nFull Aggregated Parameters:\n")
                f.write(json.dumps(aggregated_sums, indent=4))  # Pretty JSON format
                f.write("\n---\n")
            

            # Replace the old GlobalModelWeights if it exists
            global_weight = db.query(GlobalModelWeights).filter_by(session_id=session_id).first()
            if global_weight:
                global_weight.weights = aggregated_sums
                global_weight.updated_at = datetime.now()  # Add this field if not present already
                print(f"Updated existing global weight for session {session_id}")
            else:
                global_weight = GlobalModelWeights(
                    session_id=session_id,
                    weights=aggregated_sums
                )
                db.add(global_weight)
                print(f"Created new global weight for session {session_id}")
            db.commit()

@temporary_router.get("/check-aggregated-sums")
def trigger_aggregation(session_id: str , round_number: int ):
    try:
        aggregate_weights_fedAvg_Neural(session_id=session_id, round_number=round_number)
        return {"message":"Aggregation is done successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


