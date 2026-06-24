from fastapi import APIRouter, status, Depends, Request, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, update
from services.parquet import process_parquet_and_save_xy
from services.test import Test
from services.Notification import (
    send_notification_for_new_session,
    send_notification_for_new_round,
)
from utilities.core.db import get_db
from services.FederatedLearning import FederatedLearning
from utilities.core.auth import role, get_current_user
from typing import Any
from fastapi import Query
from models.FederatedSession import (
    FederatedSession,
    FederatedSessionClient,
    FederatedRoundClientSubmission,
    TrainingStatus,
    ClientPermission,
)
from utilities.ml.SampleSizeEstimation import calculate_required_data_points
from db.task_crud import get_task_by_id
from models.User import User
from multiprocessing import Process
import asyncio
from pathlib import Path
import json
from datetime import datetime, timedelta
import os
from apscheduler.triggers.date import DateTrigger
from apscheduler.schedulers.background import BackgroundScheduler

from schemas.user import ClientSessionStatusSchema
from schemas.federated import (
    CreateFederatedLearning,
    ClientFederatedResponse,
    ClientModelIdResponse,
    ClientReceiveParameters,
)
from utilities.constant.enums import FederatedSessionLogTag

federated_router = APIRouter()
federated_manager = FederatedLearning()
scheduler = BackgroundScheduler()
scheduler.start()


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
                    row,
                )
            )
        )
        for row in sessions
    ]


@federated_router.get(
    "/client/participated_sessions", response_model=list[ClientSessionStatusSchema]
)
def get_participated_sessions(
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
                        "session_id",
                        "curr_round",
                        "max_round",
                        "session_price",
                        "training_status",
                        "client_status",
                    ],
                    row,
                )
            )
        )
        for row in sessions
    ]


@federated_router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: int,
    db: Session = Depends(get_db),
):
    """Get complete status of a federated learning session"""
    return federated_manager.get_combined_session_status(session_id, db)


@federated_router.get("/federated-sessions-stats")
def get_federated_session_stats():
    stats = federated_manager.get_session_stats()
    return {
        "total_sessions": stats["total"],
        "completed_sessions": stats["completed"],
        "active_sessions": stats["active"],
    }


@federated_router.get("/get-all-federated-sessions")
def get_all_federated_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(6, ge=1, le=100),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    training_status: str | None = Query(None),
    search: str | None = Query(None),
):
    result = federated_manager.get_all(
        page=page,
        per_page=per_page,
        sort_order=sort_order,
        training_status=training_status,
        search=search,
    )

    formatted_sessions = [
        {
            "id": session_id,
            "training_status": ts,
            "name": federated_info.get("organisation_name"),
            "server_filename": federated_info.get("server_filename"),
            "created_at": createdAt,
            "curr_round": curr_round,
            "total_rounds": federated_info.get("no_of_rounds"),
        }
        for session_id, ts, federated_info, createdAt, curr_round in result["sessions"]
    ]

    return {
        "data": formatted_sessions,
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "total_pages": result["total_pages"],
        "filters": {
            "sort_order": sort_order,
            "training_status": training_status,
            "search": search,
        },
    }


def get_baseline_stats_from_task(db: Session, task_id: int):
    task = get_task_by_id(db, task_id)
    if not task:
        raise ValueError("Task not found!")
    metric_name = task.metric
    if not task.benchmark or metric_name not in task.benchmark:
        raise ValueError(f"No benchmark data found for metric '{metric_name}'")
    benchmark_data = task.benchmark.get(metric_name)
    if not benchmark_data:
        raise ValueError(f"Metric '{metric_name}' not found in benchmark")
    baseline_mean = benchmark_data.get("std_mean")
    baseline_std = benchmark_data.get("std_dev")
    if baseline_mean is None or baseline_std is None:
        raise ValueError(f"Incomplete benchmark data for '{metric_name}'")

    return baseline_mean, baseline_std

def fetch_benchmark_and_calculate_price(
    session_data: FederatedSession, db: Session
) -> float:

    task_id = int(session_data.federated_info.get("task_id"))
    baseline_mean, baseline_std = get_baseline_stats_from_task(db, task_id)

    try:
        new_mean = float(session_data.federated_info.get("expected_std_mean"))
        new_std = float(session_data.federated_info.get("expected_std_deviation"))
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Expected results must contain valid float values for std_mean and std_deviation."
        ) from e

    if new_mean is None or new_std is None:
        raise ValueError(
            "New model metrics (std_mean and std_deviation) are missing in session data."
        )

    # Extract num_predictors strictly from input_shape
    model_config = session_data.federated_info

    if not model_config:
        raise ValueError("Model Config is missing in model_info.")

    # Calculate the required data points (price)
    price = calculate_required_data_points(
        model_config, baseline_mean, baseline_std, new_mean, new_std
    )
    return price if price is not None else 100




async def _start_training_internal(session_id: int, db: Session):
    """Internal async function for starting training"""
    print("Starting training")
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        federated_manager.log_event(
            session_id, f"Session {session_id} not found", FederatedSessionLogTag.ERROR
        )
        return
    if session.training_status != TrainingStatus.ACCEPTING_CLIENTS:
        federated_manager.log_event(
            session_id,
            f"Session {session_id} is not in the correct state to start training. Current state: {session.training_status}",
            FederatedSessionLogTag.ERROR,
        )
        return
    if len(session.clients) <= int(os.getenv("MIN_CLIENTS_FOR_TRAINING", 2)):
        federated_manager.log_event(
            session_id,
            f"Session {session_id} has not enough clients to start training. Current clients: {len(session.clients)}",
            FederatedSessionLogTag.ERROR,
        )
        return
    session.training_status = TrainingStatus.STARTED
    federated_manager.log_event(
        session_id, f"Training started", FederatedSessionLogTag.TRAINING
    )
    db.commit()
    db.refresh(session)
    await send_notification_for_new_round(
        {
            "session_id": session_id,
            "round_number": 1,
            "metrics_report": {},
        }
    )
    # Start training


def start_training_sync(session_id: int):
    """Synchronous wrapper for start_training to be used with APScheduler"""
    from utilities.core.db import SessionLocal
    db = SessionLocal()
    try:
        asyncio.run(_start_training_internal(session_id, db))
    finally:
        db.close()


@federated_router.get("/force-start-training")
async def start_training_endpoint(
    request: Request, session_id: int, db: Session = Depends(get_db)
):
    """Endpoint to manually force star  t training"""
    await _start_training_internal(session_id, db)
    return {"message": "Training started successfully"}


@federated_router.post("/create-federated-session")
async def create_federated_session(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client")),
):
    print("Creating federated session")
    # Remove empty layers
    # federated_details.model_info["layers"] = [
    #     layer
    #     for layer in federated_details.model_info["layers"]
    #     if layer.get("layer_type")
    # ]
    if not request.client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client not found",
        )
    session: FederatedSession = federated_manager.create_federated_session(
        current_user, federated_details, request.client.host, db
    )
    print("Federated session created")
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

    federated_manager.log_event(
        session.id,
        "Fetching benchmark stats and calculating training price",
        FederatedSessionLogTag.PRICE_NEGOTIATION,
    )





    required_data_points = fetch_benchmark_and_calculate_price(session, db)
    federated_manager.log_event(
        session.id,
        f"Calculated training price as {required_data_points} data points",
        FederatedSessionLogTag.PRICE_NEGOTIATION,
    )

    # Store the calculated price in the session
    federated_manager.log_event(
        session.id,
        f"Storing calculated price in session {session.id}",
        FederatedSessionLogTag.PRICE_NEGOTIATION,
    )
    federated_session = db.query(FederatedSession).filter_by(id=session.id).first()
    if federated_session:
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
        federated_manager.log_event(
            session.id,
            "Price successfully stored in session",
            FederatedSessionLogTag.PRICE_NEGOTIATION,
        )
    else:
        error_msg = f"FederatedSession with ID {session.id} not found."
        federated_manager.log_event(
            session.id, f"{error_msg}", FederatedSessionLogTag.ERROR
        )
        return

    return {
        "message": "Federated Session has been created!",
        "session_id": session.id,
        "price": required_data_points,
    }


@federated_router.get("/get-federated-session/{session_id}")
async def get_federated_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client", "admin")),
):
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    session.no_of_clients = len(session.clients)

    # Calculate client_status for the current user
    # Find if the current user has a FederatedSessionClient entry
    client = next(
        (client for client in session.clients if client.user_id == current_user.id),
        None,
    )

    # Add client_status directly to the session object
    session.client_status = client.status if client else -1

    return session


@federated_router.post("/submit-client-price-acceptance")
async def submit_client_price_response(
    client_response: ClientFederatedResponse,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    """
    decision : 1 means client accepts the price, -1 means client rejects the price
    training_status = 1 means the training process should start
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

            if session.training_status != TrainingStatus.PRICE_NEGOTIATION:
                return {
                    "success": False,
                    "message": "Training is not in price negotiation state",
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
                    f"Admin Accepted the price. Waiting for wait_time to be set.",
                    FederatedSessionLogTag.PRICE_NEGOTIATION,
                )
                # Keep status as PRICE_NEGOTIATION - will be updated to ACCEPTING_CLIENTS after wait_time is submitted
                # Set price_accepted flag in federated_info (assign new dict so SQLAlchemy persists the change)
                federated_info = dict(federated_session.federated_info or {})
                federated_info["price_accepted"] = True
                federated_session.federated_info = federated_info
                message = (
                    "Thank you for accepting the price. Please set the training start time."
                )

                # Add clients who already have permission for this task
                # TEMPORARILY DISABLED FOR SINGLE-CLIENT TESTING
                # TODO: Re-enable this for multi-client production usage
                # try:
                #     task_id = int(session.federated_info.get("task_id"))
                #     if task_id:
                #         # Find all clients who have permission for this task
                #         clients_with_permission = (
                #             db.query(ClientPermission)
                #             .filter_by(task_id=task_id, permission=True)
                #             .all()
                #         )

                #         added_clients_count = 0
                #         for client_permission in clients_with_permission:
                #             # Check if client is not already in this session
                #             existing_client = (
                #                 db.query(FederatedSessionClient)
                #                 .filter_by(
                #                     session_id=session_id,
                #                     user_id=client_permission.user_id,
                #                 )
                #                 .first()
                #             )

                #             if not existing_client:
                #                 # Add client to the session
                #                 federated_session_client = FederatedSessionClient(
                #                     user_id=client_permission.user_id,
                #                     session_id=session_id,
                #                     status=0,  # JOINED status
                #                     ip="auto-added",  # Placeholder IP for auto-added clients
                #                 )
                #                 db.add(federated_session_client)
                #                 added_clients_count += 1

                #         if added_clients_count > 0:
                #             federated_manager.log_event(
                #                 session_id,
                #                 f"Automatically added {added_clients_count} clients with existing task permissions",
                #                 FederatedSessionLogTag.INFO,
                #             )
                # except (ValueError, TypeError) as e:
                #     # Log error but don't fail the request if task_id is invalid
                #     print(f"Error adding clients with existing permissions: {e}")

                process_parquet_and_save_xy(
                    session.federated_info["server_filename"],
                    session_id,
                    session.federated_info["input_columns"],
                    session.federated_info["output_columns"],
                )
                await send_notification_for_new_session(
                    "New session created with session id: " + str(session_id)
                )
                # Do NOT schedule cron job here - it will be scheduled after wait_time is submitted
                federated_manager.log_event(
                    session_id,
                    "Price accepted. Waiting for admin to set training start time.",
                    FederatedSessionLogTag.PRICE_NEGOTIATION,
                )
            elif decision == 0:
                federated_manager.log_event(
                    session_id,
                    f"Admin rejected the price updating training status = -1",
                    FederatedSessionLogTag.PRICE_NEGOTIATION,
                )
                federated_session.training_status = (
                    TrainingStatus.CANCELLED
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


@federated_router.post("/submit-wait-time")
async def submit_wait_time(
    request_data: dict,
    current_user: User = Depends(role("client")),
    db: Session = Depends(get_db),
):
    """
    Endpoint to submit wait_time after price acceptance.
    Updates status to ACCEPTING_CLIENTS and schedules cron job.
    """
    try:
        session_id = request_data.get("session_id")
        wait_time = request_data.get("wait_time")

        if session_id is None or wait_time is None:
            raise HTTPException(
                status_code=400,
                detail="session_id and wait_time are required",
            )

        if wait_time < 0:
            raise HTTPException(
                status_code=400,
                detail="wait_time must be 0 or greater",
            )

        session = federated_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404, detail="Federated session not found"
            )

        # Only admin can set wait_time
        if session.admin_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Unauthorized. Only the admin can set wait_time.",
            )

        # Validate session is in PRICE_NEGOTIATION status
        if session.training_status != TrainingStatus.PRICE_NEGOTIATION:
            raise HTTPException(
                status_code=400,
                detail=f"Session is not in PRICE_NEGOTIATION status. Current status: {session.training_status}",
            )

        # Validate that price has been accepted
        federated_info = session.federated_info
        if not isinstance(federated_info, dict):
            federated_info = {}
        
        if not federated_info.get("price_accepted"):
            raise HTTPException(
                status_code=400,
                detail="Price has not been accepted yet. Please accept the price first.",
            )

        # Get session from database to update
        federated_session = (
            db.query(FederatedSession).filter_by(id=session_id).first()
        )
        if not federated_session:
            raise HTTPException(
                status_code=404, detail="Federated session not found"
            )

        # Update federated_info with wait_time
        federated_info["wait_time"] = wait_time
        federated_session.federated_info = federated_info

        # Update training_status to ACCEPTING_CLIENTS
        federated_session.training_status = TrainingStatus.ACCEPTING_CLIENTS

        # Commit changes
        db.commit()
        db.refresh(federated_session)

        # Log event
        federated_manager.log_event(
            session_id,
            f"Admin set wait_time to {wait_time} minutes. Training will start after {wait_time} minutes.",
            FederatedSessionLogTag.PRICE_NEGOTIATION,
        )

        # Schedule cron job with provided wait_time
        trigger = DateTrigger(
            run_date=datetime.now() + timedelta(minutes=wait_time)
        )
        scheduler.add_job(start_training_sync, trigger, args=[session_id])

        federated_manager.log_event(
            session_id,
            f"Cron job scheduled to start training in {wait_time} minutes.",
            FederatedSessionLogTag.PRICE_NEGOTIATION,
        )

        return {
            "success": True,
            "message": f"Training start time set successfully. Training will start in {wait_time} minutes.",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in submit_wait_time: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        )


@federated_router.post("/accept-training")
async def accept_training(
    client_response: ClientFederatedResponse,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client")),
):
    """
    This endpoint is used to accept or reject the training session by other clients
    client_response : session_id and decision
    decision : 1 means client accepts and 0 means client rejects
    """
    session_id = client_response.session_id
    decision = client_response.decision
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.training_status != TrainingStatus.ACCEPTING_CLIENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clients are not being accepted yet",
        )
    if decision == 0:
        return {
            "success": True,
            "message": "Your decision to decline participation in the training session has been recorded. Thank you for your response.",
        }
    if decision != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid decision value. Must be 1 (accept) or 0 (reject).",
        )
    client = (
        db.query(FederatedSessionClient)
        .filter_by(session_id=session_id, user_id=current_user.id)
        .first()
    )
    if client:
        return {"success": True, "message": "Client Decision has already been saved"}
    if not request.client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client not found",
        )
    if not client:
        federated_session_client = FederatedSessionClient(
            user_id=current_user.id,
            session_id=session_id,
            status=0,
            ip=request.client.host,
        )
        db.add(federated_session_client)

        # Add client permission for the corresponding task
        try:
            task_id = int(session.federated_info.get("task_id"))
            if task_id:
                # Check if permission already exists
                existing_permission = (
                    db.query(ClientPermission)
                    .filter_by(user_id=current_user.id, task_id=task_id)
                    .first()
                )

                if not existing_permission:
                    # Create new client permission
                    client_permission = ClientPermission(
                        user_id=current_user.id, task_id=task_id, permission=True
                    )
                    db.add(client_permission)
        except (ValueError, TypeError) as e:
            # Log error but don't fail the request if task_id is invalid
            print(f"Error creating client permission: {e}")

        db.commit()
    return {"success": True, "message": "Client Decision has been saved"}


@federated_router.get("/get-weights/{session_id}")
def get_weights(session_id: int, db: Session = Depends(get_db)):
    """
    Client can receive the model parameters / weights and start training
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


async def aggregate_and_test_weights(session_id: int, round_number: int, db: Session):
    """Background task to aggregate weights and run tests"""
    try:
        # Aggregate weights using FedAvg
        federated_manager.aggregate_weights_fedAvg_Neural(session_id, round_number)

        test = Test(session_id)

        federated_manager.log_event(
            session_id, f"Initialized test unit.", FederatedSessionLogTag.INFO
        )
        print("Getting latest global weights")
        # print(federated_manager.get_latest_global_weights(session_id))
        results = test.start_test(
            federated_manager.get_latest_global_weights(session_id)
        )

        federated_manager.log_event(
            session_id,
            f"Global test results: {results}",
            FederatedSessionLogTag.TEST_RESULTS,
        )

        # Reset client_parameters to an empty JSON object
        federated_manager.clear_client_parameters(session_id, round_number)

        federated_manager.log_event(
            session_id,
            f"Client parameters reset after Round {round_number}.",
            FederatedSessionLogTag.TRAINING,
        )

        session_data = (
            db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
        )
        if session_data:
            session_data.curr_round = round_number + 1
            session_data.no_of_recieved_weights = 0
            # session_data.no_of_left_clients = 0

            if (
                session_data.curr_round
                == session_data.federated_info["no_of_rounds"] + 1
            ):
                federated_manager.log_event(
                    session_id,
                    f"Training completed for session {session_id}.",
                    FederatedSessionLogTag.SUCCESS,
                )
                session_data.training_status = TrainingStatus.COMPLETED
                db.commit()
                db.refresh(session_data)
                return
            else:
                db.commit()
                db.refresh(session_data)

        # Send notification for new round
        await send_notification_for_new_round(
            {
                "session_id": session_id,
                "round_number": round_number + 1,
                "metrics_report": {},
            }
        )

    except Exception as e:
        federated_manager.log_event(
            session_id,
            f"Error in aggregate_and_test_weights: {str(e)}",
            FederatedSessionLogTag.ERROR,
        )


@federated_router.post("/send-weights")
def send_weights(
    request: ClientReceiveParameters,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(role("client")),
):
    session_id = request.session_id
    weights = request.client_parameter
    metrics_report = request.metrics_report

    try:
        # Acquire FOR UPDATE lock on the session row to serialize concurrent client submissions
        session_data = (
            db.query(FederatedSession)
            .filter(FederatedSession.id == session_id)
            .with_for_update()
            .first()
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

        with open(weights_file, "w") as f:
            json.dump(weights, f)

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
        db.flush()  # Flushes the new submission into the current transaction context

        # Since it is flushed, the count query will include this new submission
        received_weights = (
            db.query(FederatedRoundClientSubmission)
            .filter_by(session_id=session_id, round_number=round_number)
            .count()
        )

        session_data.no_of_recieved_weights = received_weights
        
        # Commit the transaction (this releases the lock and commits all changes)
        db.commit()

        # Log that client parameters were received
        federated_manager.log_event(
            session_id,
            f"Received client parameters from user {current_user.id} for round {round_number}",
            FederatedSessionLogTag.WEIGHTS_RECEIVED,
        )

        # Refresh session_data to log details and check for aggregation
        db.refresh(session_data)

        # Debug logging for aggregation condition
        total_clients = len(session_data.clients)
        print("total_clients", total_clients)
        left_clients = session_data.no_of_left_clients

        # Log detailed client information
        client_details = []
        for client in session_data.clients:
            client_details.append(
                f"User {client.user_id} (IP: {client.ip}, Status: {client.status})"
            )

        federated_manager.log_event(
            session_id,
            f"Aggregation check: {total_clients} total clients, {received_weights} received weights, {left_clients} left clients",
            FederatedSessionLogTag.INFO,
        )

        federated_manager.log_event(
            session_id,
            f"Registered clients: {', '.join(client_details)}",
            FederatedSessionLogTag.INFO,
        )

        if total_clients == received_weights + left_clients:
            federated_manager.log_event(
                session_data.id,
                f"All clients have submitted weights. Starting aggregation for round {session_data.curr_round}.",
                FederatedSessionLogTag.AGGREGATED_WEIGHTS,
            )
            federated_manager.log_event(
                session_data.id,
                f"Adding background task for aggregation and testing (round {session_data.curr_round})",
                FederatedSessionLogTag.INFO,
            )
            background_tasks.add_task(
                aggregate_and_test_weights, session_data.id, session_data.curr_round, db
            )
        else:
            federated_manager.log_event(
                session_data.id,
                f"Not all clients submitted yet. Waiting for more submissions.",
                FederatedSessionLogTag.INFO,
            )

        return {"message": "Client Parameters Received"}

    except HTTPException:
        raise
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


@federated_router.get("/training-result/{session_id}")
def get_training_result(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Fetch test results from the database for a given FederatedSession ID.
    """
    session = db.query(FederatedSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Federated session not found.")

    federated_info = session.federated_info or {}
    model_info = federated_info.get("model_info", {})
    test_metrics = model_info.get("test_metrics", [])

    server_results = {}
    raw_server_results = session.as_dict().get("results", [])
    for result in raw_server_results:
        round_number = result.get("round_number")
        metrics = result.get("metrics_report", {})

        for metric, value in metrics.items():
            if metric not in server_results:
                server_results[metric] = {}
            server_results[metric][f"round_{round_number}"] = value

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

    return {
        "session_id": session_id,
        "current_round": session.curr_round,
        "test_metrics": test_metrics,
        "server_results": server_results,
        "client_results": client_results,
    }


@federated_router.get("/download-model-parameters/{session_id}")
def download_model_parameters(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download global model parameters when training is complete (session admin only).
    """
    session = (
        db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user.id != session.admin_id:
        raise HTTPException(
            status_code=403, detail="Only the session admin can access model parameters"
        )

    if session.training_status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=403,
            detail="Model parameters are only available after training completion",
        )

    global_params_dir = Path(f"tmp/parameters/{session_id}/global/")
    global_params_file = global_params_dir / "global_weights.json"

    if not global_params_file.exists():
        raise HTTPException(status_code=404, detail="Model parameters not found")
    try:
        return FileResponse(
            str(global_params_file),
            media_type="application/json",
            filename=f"model_parameters_{session_id}.json",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving model parameters: {str(e)}"
        )
