from fastapi import APIRouter, status, Depends, Request, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, update
from utility.Notification import send_notification_for_new_session
from utility.db import get_db
from utility.FederatedLearning import FederatedLearning
from utility.auth import role, get_current_user
from utility.federated_learning import (
    fetch_benchmark_and_calculate_price,
    start_federated_learning,
)
from typing import Any
from fastapi import Query
from models.FederatedSession import (
    FederatedSession,
    FederatedSessionClient,
    FederatedRoundClientSubmission,
)
from models.User import User
from multiprocessing import Process
import asyncio
from pathlib import Path
import json
from datetime import datetime

from schemas.user import ClientSessionStatusSchema
from schema import (
    CreateFederatedLearning,
    ClientFederatedResponse,
    ClientModelIdResponse,
    ClientReceiveParameters,
)

federated_router_v2 = APIRouter()
federated_manager = FederatedLearning()


@federated_router_v2.post("/create-federated-session")
async def create_federated_session_v2(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client")),
):
    # Remove empty layers
    federated_details.fed_info.model_info["layers"] = [
        layer
        for layer in federated_details.fed_info.model_info["layers"]
        if layer.get("layer_type")
    ]
    session: FederatedSession = federated_manager.create_federated_session(
        current_user, federated_details.fed_info, request.client.host, db
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Federated session could not be created.",
        )

    federated_manager.log_event(
        session.id,
        f"Federated session created by admin {current_user.id} from {request.client.host}",
    )

    federated_manager.log_event(
        session.id, "Fetching benchmark stats and calculating training price"
    )
    required_data_points = fetch_benchmark_and_calculate_price(session, db)
    federated_manager.log_event(
        session.id, f"Calculated training price as {required_data_points} data points"
    )

    # Store the calculated price in the session
    federated_manager.log_event(
        session.id, f"Storing calculated price in session {session.id}"
    )
    federated_session = db.query(FederatedSession).filter_by(id=session.id).first()
    if federated_session:
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
        federated_manager.log_event(session.id, "Price successfully stored in session")
    else:
        error_msg = f"FederatedSession with ID {session.id} not found."
        federated_manager.log_event(session.id, f"{error_msg}")
        return

    return {
        "message": "Federated Session has been created!",
        "session_id": session.id,
        "price": required_data_points,
    }


@federated_router_v2.get("/get-federated-session/{session_id}")
async def get_federated_session_v2(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client")),
):
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return session


@federated_router_v2.post("/submit-client-price-acceptance")
def submit_client_price_response(
    client_response: ClientFederatedResponse,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    """
    decision : 1 means client accepts the price, -1 means client rejects the price
    training_status = 2 means the training process should start
    """
    try:
        session_id = client_response.session_id
        decision = client_response.decision
        session = federated_manager.get_session(session_id)
        message = ""
        if session:
            # Only admin can respond
            if session.admin_id != current_user.id:
                return {
                    "success": False,
                    "message": "Unauthorized user. Can only be accepted by the admin of this session",
                }

            if session.training_status == 2:
                return {"success": False, "message": "Training has already started"}

            federated_session = (
                db.query(FederatedSession).filter_by(id=session_id).first()
            )

            if not federated_session:
                raise HTTPException(
                    status_code=404, detail="Federated session not found"
                )
            # Update training_status based on the decision
            if decision == 1:
                federated_manager.log_event(
                    session_id, f"Admin Accepted the price updating training status = 2"
                )
                federated_session.training_status = (
                    2  # Update training_status to 2 (start training)
                )
                message = (
                    "Thank you for accepting the price. The training will start soon."
                )
                await send_notification_for_new_session(
                    "New session created with session id: " + str(session_id)
                )
            elif decision == 0:
                federated_manager.log_event(
                    session_id,
                    f"Admin rejected the price updating training status = -1",
                )
                federated_session.training_status = (
                    -1
                )  # Keep or set to a default status for rejection
                message = (
                    "Thank you for rejecting the price. The training will not start."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid decision value. Must be 1 (accept) or -1 (reject).",
                )
            # Commit changes to the database
            db.commit()
            return {"success": True, "message": message}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
