from fastapi import APIRouter, status, Depends, Request, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, update
from utility.db import get_db
from utility.FederatedLearning import FederatedLearning
from utility.auth import role, get_current_user
from utility.federated_learning import start_federated_learning
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
from constant.enums import FederatedSessionLogTag

federated_router = APIRouter()
federated_manager = FederatedLearning()


@federated_router.get(
    "/client/initiated_sessions", response_model=list[ClientSessionStatusSchema]
)
def get_initiated_jobs(
    current_user: User = Depends(role("client")), db: Session = Depends(get_db)
):
    sessions = (
        db.query(
            FederatedSession.id,
            FederatedSession.curr_round,
            FederatedSession.max_round,
            FederatedSession.session_price,
            FederatedSession.training_status,
            FederatedSessionClient.status.label("client_status"),
        )
        .outerjoin(
            FederatedSessionClient,
            (FederatedSession.id == FederatedSessionClient.session_id)
            & (FederatedSessionClient.user_id == current_user.id),
        )
        .filter(FederatedSession.admin_id == current_user.id)
        .all()
    )

    return [
        ClientSessionStatusSchema.model_validate(
            dict(
                zip(
                    [
                        "session_id",
                        "curr_round",
                        "max_round",
                        "session_price",
                        "training_status",
                        "client_status",
                    ],
                    session,
                )
            )
        )
        for session in sessions
    ]


@federated_router.get(
    "/client/participated_sessions", response_model=list[ClientSessionStatusSchema]
)
def get_participated_sessions(
    current_user: User = Depends(role("client")), db: Session = Depends(get_db)
):
    sessions = (
        db.query(
            FederatedSession.curr_round,
            FederatedSession.max_round,
            FederatedSession.session_price,
            FederatedSession.training_status,
            FederatedSessionClient.status.label("client_status"),
        )
        .join(
            FederatedSessionClient,
            FederatedSession.id == FederatedSessionClient.session_id,
        )
        .filter(FederatedSessionClient.user_id == current_user.id)
        .all()
    )
    return [
        ClientSessionStatusSchema.model_validate(
            dict(
                zip(
                    [
                        "curr_round",
                        "max_round",
                        "session_price",
                        "training_status",
                        "client_status",
                    ],
                    session,
                )
            )
        )
        for session in sessions
    ]


@federated_router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: int,
    db: Session = Depends(get_db),
):
    """Get complete status of a federated learning session"""
    # Verify user has access to the session first
    return federated_manager.get_combined_session_status(session_id, db)


def run_async_in_process(coroutine_func, *args, **kwargs):
    """Helper function to run async functions in a process"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coroutine_func(*args, **kwargs))
    finally:
        loop.close()


@federated_router.post("/create-federated-session")
async def create_federated_session(
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
        FederatedSessionLogTag.INFO,
    )

    try:
        # background_tasks.add_task(start_federated_learning, federated_manager, current_user, session, db)
        # background_tasks.add_task(start_federated_learning_wrapper, federated_manager, current_user, session, db)
        process = Process(
            target=run_async_in_process,
            args=(
                start_federated_learning,
                federated_manager,
                current_user,
                session,
                db,
            ),
        )
        process.start()
        federated_manager.add_process(session.id, process)
        federated_manager.log_event(
            session.id,
            "Background task for federated learning started",
            FederatedSessionLogTag.TRAINING,
        )
    except Exception as e:
        federated_manager.log_event(
            session.id,
            f"Error starting background task: {str(e)}",
            FederatedSessionLogTag.ERROR,
        )
        return {"message": "An error occurred while starting federated learning."}

    return {"message": "Federated Session has been created!", "session_id": session.id}


@federated_router.get("/get-all-federated-sessions")
def get_all_federated_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(6, ge=1, le=100),
):
    all_sessions = [
        {
            "id": id,
            "training_status": training_status,
            "name": federated_info.get("organisation_name"),
            "created_at": createdAt,
        }
        for [
            id,
            training_status,
            federated_info,
            createdAt,
        ] in federated_manager.get_all()
    ]

    # Calculate pagination
    total = len(all_sessions)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_sessions = all_sessions[start:end]

    return {
        "data": paginated_sessions,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@federated_router.get("/get-federated-session/{session_id}")
def get_federated_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client", "admin")),
):
    try:
        federated_session_data = (
            db.query(FederatedSession).filter_by(id=session_id).first()
        )
        if not federated_session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        client = next(
            (
                client
                for client in federated_session_data.clients
                if client.user_id == current_user.id
            ),
            None,
        )

        federated_response = {
            "federated_info": federated_session_data.federated_info,
            "training_status": federated_session_data.training_status,
            "client_status": client.status if client else -1,
            "session_price": federated_session_data.session_price,
        }

        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@federated_router.post("/submit-client-price-acceptance-response")
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
        if session:
            # Only admin can respond
            if session.admin_id != current_user.id:
                return {
                    "success": False,
                    "message": "Only the admin of this session can respond",
                }

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
                    session_id,
                    f"Admin Accepted the price updating training status = 2",
                    FederatedSessionLogTag.PRICE_NEGOTIATION,
                )
                federated_session.training_status = (
                    2  # Update training_status to 2 (start training)
                )
            elif decision == 0:
                federated_manager.log_event(
                    session_id,
                    f"Admin rejected the price updating training status = -1",
                    FederatedSessionLogTag.PRICE_NEGOTIATION,
                )
                federated_session.training_status = (
                    -1
                )  # Keep or set to a default status for rejection
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid decision value. Must be 1 (accept) or -1 (reject).",
                )
            # Commit changes to the database
            db.commit()
            return {"success": True, "message": "Training status updated successfully"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@federated_router.post("/submit-client-training-acceptance-response")
def submit_client_federated_response(
    client_response: ClientFederatedResponse,
    request: Request,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    """
    decision : 1 means client accepts and 0 means rejects
    """
    session_id = client_response.session_id
    decision = client_response.decision
    if decision == 0:
        return {
            "success": True,
            "message": "Your decision to decline participation in the training session has been recorded. Thank you for your response.",
        }
    session = federated_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    client = (
        db.query(FederatedSessionClient)
        .filter_by(session_id=session_id, user_id=current_user.id)
        .first()
    )
    if not client:
        federated_session_client = FederatedSessionClient(
            user_id=current_user.id,
            session_id=session_id,
            status=0,
            ip=request.client.host,
        )
        db.add(federated_session_client)
        db.commit()
    return {"success": True, "message": "Client Decision has been saved"}


@federated_router.post("/client-initialize-model")
def client_initialize_model(
    request: ClientModelIdResponse,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    """
    Client has initialized the model and notifies the server.
    Status is updated from 0 to 1.
    """
    session_id = request.session_id
    db.execute(
        update(FederatedSessionClient)
        .where(
            and_(
                FederatedSessionClient.user_id == current_user.id,
                FederatedSessionClient.session_id == session_id,
            )
        )
        .values(
            status=1,
        )
    )
    db.commit()
    return {"message": "Client status updated to 1 (initialized model)"}


@federated_router.get("/get-model-parameters/{session_id}")
def get_model_parameters(session_id: int, db: Session = Depends(get_db)):
    """
    Client have received the model parameters and waiting for server to start training
    """
    # Path to check for global parameters
    global_params_dir = Path(f"tmp/parameters/{session_id}/global/")
    global_params_file = global_params_dir / "global_weights.json"

    # Check if global parameters file exists
    if global_params_file.exists():
        try:
            # Load global parameters from file
            with open(global_params_file, "r") as f:
                global_parameters = json.load(f)
            is_first = 0
        except Exception as e:
            # If file exists but can't be read, treat as first round
            is_first = 1
            global_parameters = {}
    else:
        # No global parameters file exists - first round
        is_first = 1
        global_parameters = {}

    response_data = {"global_parameters": global_parameters, "is_first": is_first}

    return response_data


@federated_router.post("/receive-client-parameters")
def receive_client_parameters(
    request: ClientReceiveParameters,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    session_id = request.session_id
    client_parameter = request.client_parameter
    metrics_report = request.metrics_report

    session_data = (
        db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    )

    if not session_data:
        raise HTTPException(
            status_code=404, detail=f"Federated Session with ID {session_id} not found!"
        )

    round_number = session_data.curr_round

    # Check if a submission already exists for this user, session, and round
    submission = (
        db.query(FederatedRoundClientSubmission)
        .filter_by(
            session_id=session_id, user_id=current_user.id, round_number=round_number
        )
        .first()
    )

    if submission:
        federated_manager.log_event(
            session_id,
            f"Client parameters for this round {round_number} already submitted.",
            FederatedSessionLogTag.ERROR,
        )
        raise HTTPException(
            status_code=400,
            detail="Client parameters for this round already submitted.",
        )

    # Create directory structure
    base_dir = Path(f"tmp/parameters/{session_id}")
    local_dir = base_dir / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    weights_file = local_dir / f"{current_user.id}.json"
    metadata_file = local_dir / f"{current_user.id}_metadata.json"

    try:
        with open(weights_file, "w") as f:
            json.dump(client_parameter, f)

        metadata = {
            "submission_time": datetime.utcnow().isoformat(),
            "user_id": current_user.id,
            "round_number": round_number,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create new submission entry
        submission = FederatedRoundClientSubmission(
            session_id=session_id,
            user_id=current_user.id,
            round_number=round_number,
            metrics_report=metrics_report,
        )
        db.add(submission)
        db.flush()  # Ensures submission.id is available before committing
        db.commit()
        federated_manager.log_event(
            session_id,
            f"Received client parameters from user {current_user.id} for round {round_number}",
            FederatedSessionLogTag.WEIGHTS_RECEIVED,
        )
        return {"message": "Client Parameters Received"}
    except Exception as e:
        db.rollback()
        federated_manager.log_event(
            session_id,
            f"Error receiving client parameters: {str(e)}",
            FederatedSessionLogTag.ERROR,
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing client parameters: {str(e)}"
        )


@federated_router.get("/training-result/{session_id}")
def get_training_result(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Fetch test results directly from the database for a given FederatedSession ID.
    """
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Federated session not found.")

    # Get test metrics from federated_info
    federated_info = session.federated_info or {}
    model_info = federated_info.get("model_info", {})
    test_metrics = model_info.get("test_metrics", [])

    # Get Server Round Results
    server_results = {}
    print("Model Info", model_info)
    print("Session: ", session)
    raw_server_results = session.as_dict().get("results", [])
    print("Raw server results: ", raw_server_results)
    print("Test metrics: ", test_metrics)
    for result in raw_server_results:
        round_number = result.get("round_number")
        metrics = result.get("metrics_report", {})

        for metric, value in metrics.items():
            if metric not in server_results:
                server_results[metric] = {}
            server_results[metric][f"round_{round_number}"] = value

    # Restructure client results
    client_results = {}
    if current_user:
        submissions = (
            db.query(FederatedRoundClientSubmission)
            .filter_by(session_id=session_id, user_id=current_user.id)
            .order_by(FederatedRoundClientSubmission.round_number)
            .all()
        )

        for submission in submissions:
            data = submission.as_dict()
            round_number = data.get("round_number")
            metrics = data.get("metrics_report", {})

            for metric, value in metrics.items():
                if metric not in client_results:
                    client_results[metric] = {}
                client_results[metric][f"round_{round_number}"] = value

    response = {
        "session_id": session_id,
        "current_round": session.curr_round,
        "test_metrics": test_metrics,
        "server_results": server_results,
        "client_results": client_results,
    }

    return response


@federated_router.get("/download-model-parameters/{session_id}")
def get_model_parameters(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get global model parameters with admin check.
    If training is complete and user is admin, allows download.
    """
    # Get the session first
    session = (
        db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify the current user is the session admin
    if current_user.id != session.admin_id:
        raise HTTPException(
            status_code=403, detail="Only the session admin can access model parameters"
        )

    # Verify training is complete
    if session.training_status != "COMPLETED":  # Assuming 5 means complete
        raise HTTPException(
            status_code=403,
            detail="Model parameters are only available after training completion",
        )

    # Path to check for global parameters
    global_params_dir = Path(f"tmp/parameters/{session_id}/global/")
    global_params_file = global_params_dir / "global_weights.json"

    # Check if global parameters file exists
    if not global_params_file.exists():
        raise HTTPException(status_code=404, detail="Model parameters not found")
    try:
        # Return as downloadable file
        return FileResponse(
            str(global_params_file),
            media_type="application/json",
            filename=f"model_parameters_{session_id}.json",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving model parameters: {str(e)}"
        )
