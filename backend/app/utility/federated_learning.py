from datetime import datetime
from typing import Dict
from requests import session
from sqlalchemy import Null
from models.FederatedSession import (
    FederatedSession,
    FederatedSessionClient,
    FederatedRoundClientSubmission,
)
from utility.FederatedLearning import FederatedLearning
from models import User
import asyncio
import os
import json
from utility.db import engine
from sqlalchemy.orm import Session
from utility.notification import (
    add_notifications_for,
    add_notifications_for_user,
    add_notifications_for_recently_active_users,
)
from utility.SampleSizeEstimation import calculate_required_data_points
from crud.task_crud import (
    get_task_by_id,
)
from constant.message_type import MessageType
import random
from utility.test import Test
from helpers.federated_services import process_parquet_and_save_xy
import shutil
from constant.enums import FederatedSessionLogTag


def save_weights_to_file(weights: dict, filename: str):
    """Save the given weights dictionary to a JSON file."""
    if weights is None:
        weights = {}

    dirpath = os.path.dirname(filename)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    with open(filename, "a") as f:
        json.dump(weights, f, indent=4)


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


async def start_federated_learning(
    federated_manager: FederatedLearning,
    user: User,
    session_data: FederatedSession,
    db: Session,
):
    """
    Background task to manage federated learning rounds.

    This function runs in the background, waiting for client responses before proceeding with each round
    of federated learning.

    Each round consists of:
    1. Setting the current round number (`curr_round`) in the server.
    2. Printing round information.
    3. Receiving parameters from clients.
    4. Aggregating weights using federated averaging with neural networks.

    """

    # Fetch benchmark stats and calculate required data points (price)
    federated_manager.log_event(
        session_data.id,
        "Starting federated learning process",
        FederatedSessionLogTag.INFO,
    )

    federated_manager.log_event(
        session_data.id,
        "Fetching benchmark stats and calculating training price",
        FederatedSessionLogTag.PRIZE_NEGOTIATION,
    )
    required_data_points = fetch_benchmark_and_calculate_price(session_data, db)
    federated_manager.log_event(
        session_data.id,
        f"Calculated training price as {required_data_points} data points",
        FederatedSessionLogTag.PRIZE_NEGOTIATION,
    )

    # Store the calculated price in the session
    federated_manager.log_event(
        session_data.id,
        f"Storing calculated price in session {session_data.id}",
        FederatedSessionLogTag.PRIZE_NEGOTIATION,
    )
    federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
    if federated_session:
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
        federated_manager.log_event(
            session_data.id,
            "Price successfully stored in session",
            FederatedSessionLogTag.PRIZE_NEGOTIATION,
        )
    else:
        error_msg = f"FederatedSession with ID {session_data.id} not found."
        federated_manager.log_event(
            session_data.id, f"{error_msg}", FederatedSessionLogTag.ERROR
        )
        return

    # Send the price to the client and wait for approval
    federated_manager.log_event(
        session_data.id,
        "Waiting for client price confirmation",
        FederatedSessionLogTag.PRIZE_NEGOTIATION,
    )
    approved = await wait_for_price_confirmation(federated_manager, session_data.id, db)

    if not approved:
        federated_session = (
            db.query(FederatedSession).filter_by(id=session_data.id).first()
        )
        if not federated_session:
            raise Exception(f"FederatedSession with ID {session_data.id} not found.")
        federated_session.training_status = -1
        db.commit()
        federated_manager.log_event(
            session_data.id,
            f"Client {user.id} declined the price. Training aborted.",
            FederatedSessionLogTag.PRIZE_NEGOTIATION,
        )
        return

    federated_manager.log_event(
        session_data.id,
        f"Client {user.id} accepted the price. Training starts.",
        FederatedSessionLogTag.TRAINING,
    )

    message = {
        "type": "new-session",
        "message": "New Federated Session Avaliable!",
        "session_id": session_data.id,
    }

    # Make alert in Client side
    federated_manager.log_event(
        session_data.id,
        "Sending notifications to active users",
        FederatedSessionLogTag.INFO,
    )
    add_notifications_for_recently_active_users(
        db=db,
        message=message,
        valid_until=session_data.federated_info.wait_time,
        excluded_users=[user],
    )
    federated_manager.log_event(
        session_data.id,
        f"Notification:new-session is sent to all users.",
        FederatedSessionLogTag.INFO,
    )

    # Wait for client confirmation of interest
    await wait_for_client_confirmation(federated_manager, session_data.id, db)
    federated_manager.log_event(
        session_data.id,
        "Client confirmation process completed.",
        FederatedSessionLogTag.INFO,
    )

    # Send Model Configurations to interested clients and wait for their confirmation
    success = await send_model_configs_and_wait_for_confirmation(
        federated_manager, session_data.id
    )
    if success:
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = (
                db.query(FederatedSession).filter_by(id=session_data.id).first()
            )
            if not federated_session:
                raise Exception(
                    f"FederatedSession with ID {session_data.id} not found."
                )
            federated_session.training_status = 4
            db.commit()
        federated_manager.log_event(
            session_data.id,
            "All interested clients confirmed model configuration",
            FederatedSessionLogTag.SUCCESS,
        )
    else:
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = (
                db.query(FederatedSession).filter_by(id=session_data.id).first()
            )
            if not federated_session:
                raise Exception(
                    f"FederatedSession with ID {session_data.id} not found."
                )
            federated_session.training_status = -1
            db.commit()
        federated_manager.log_event(
            session_data.id,
            "Timeout: Not all clients confirmed model configuration",
            FederatedSessionLogTag.ERROR,
        )
        return

    #############################################
    # code used to get instance of testing unit
    # Here Input has to be taken in future for the metrics
    test = Test(session_data.id)
    federated_manager.log_event(
        session_data.id, f"Initialized test unit.", FederatedSessionLogTag.INFO
    )

    # Download data from Hadoop
    federated_info = session_data.federated_info
    dataset_info = federated_info.get("dataset_info")
    server_filename = dataset_info.get("server_filename")
    output_columns = dataset_info.get("output_columns")
    process_parquet_and_save_xy(server_filename, session_data.id, output_columns)

    # Start Training
    federated_manager.log_event(
        session_data.id,
        f"Starting training with {session_data.max_round} rounds",
        FederatedSessionLogTag.TRAINING,
    )
    for i in range(1, session_data.max_round + 1):
        session_data = federated_manager.get_session(session_data.id)
        federated_manager.log_event(
            session_data.id, f"Starting round {i}.", FederatedSessionLogTag.TRAINING
        )
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = (
                db.query(FederatedSession).filter_by(id=session_data.id).first()
            )
            if not federated_session:
                raise Exception(
                    f"FederatedSession with ID {session_data.id} not found."
                )
            federated_session.curr_round = i
            db.commit()
        federated_manager.log_event(
            session_data.id,
            f"Round {i} marked in database",
            FederatedSessionLogTag.TRAINING,
        )

        await send_training_signal_and_wait_for_clients_training(
            federated_manager, session_data.id
        )
        # Aggregate
        federated_manager.log_event(
            session_data.id,
            f"Performing aggregation.",
            FederatedSessionLogTag.AGGREGATED_WEIGHTS,
        )

        federated_manager.aggregate_weights_fedAvg_Neural(session_data.id, i)

        federated_manager.log_event(
            session_data.id,
            f"Aggregation is done",
            FederatedSessionLogTag.AGGREGATED_WEIGHTS,
        )

        ################ Testing start
        results = test.start_test(
            federated_manager.get_latest_global_weights(session_data.id)
        )
        federated_manager.log_event(
            session_data.id,
            f"Global test results: {results}",
            FederatedSessionLogTag.TEST_RESULTS,
        )

        # Reset client_parameters to an empty JSON object
        federated_manager.clear_client_parameters(session_data.id, i)

        federated_manager.log_event(
            session_data.id,
            f"Client parameters reset after Round {i}.",
            FederatedSessionLogTag.TRAINING,
        )

    # Deleted data folder after training is complete
    local_dir = os.path.join(os.getcwd(), "data")
    shutil.rmtree(local_dir)

    with Session(engine) as db:
        # Update the training status of the session
        federated_session = (
            db.query(FederatedSession).filter_by(id=session_data.id).first()
        )
        if not federated_session:
            raise Exception(f"FederatedSession with ID {session_data.id} not found.")
        federated_session.training_status = 5
        db.commit()
    federated_manager.log_event(
        session_data.id,
        f"Training completed. Test results saved.",
        FederatedSessionLogTag.TEST_RESULTS,
    )


async def wait_for_price_confirmation(
    federated_manager: FederatedLearning,
    session_id: str,
    db: Session,
    timeout: int = 3000,
):
    """
    Asynchronously waits for the client to accept the price before proceeding with federated learning.

    Args:
        federated_manager (FederatedLearning): The federated learning manager handling sessions.
        session_id (str): The ID of the federated learning session.
        timeout (int): Maximum time (in seconds) to wait for the price confirmation. Default is 5 minutes.

    Returns:
        bool: True if the client accepted the price, False if timeout occurred.
    """
    start_time = asyncio.get_event_loop().time()  # Get async time to prevent blocking
    session_data = federated_manager.get_session(int(session_id))
    federated_manager.log_event(
        session_data.id,
        f"Starting price confirmation wait (timeout: {timeout}s)",
        FederatedSessionLogTag.PRIZE_NEGOTIATION,
    )

    while True:
        session_data = federated_manager.get_session(int(session_id))

        if session_data.training_status == 2:  # Status 2 means price was accepted
            federated_manager.log_event(
                session_data.id,
                f"Client accepted the price for session {session_id}",
                FederatedSessionLogTag.PRIZE_NEGOTIATION,
            )
            return True

        # Check for timeout
        if asyncio.get_event_loop().time() - start_time > timeout:
            federated_manager.log_event(
                session_data.id,
                f"Timeout: Client did not confirm price within {timeout}s",
                FederatedSessionLogTag.PRIZE_NEGOTIATION,
            )
            return False

        federated_manager.log_event(
            session_data.id,
            "Waiting for client price confirmation.",
            FederatedSessionLogTag.PRIZE_NEGOTIATION,
        )
        await asyncio.sleep(5)  # Non-blocking sleep


async def wait_for_client_confirmation(
    federated_manager: FederatedLearning,
    session_id: int,
    db: Session,
    timeout: int = 60,  # 60 second timeout here
):
    """
    Waits for a fixed timeout period for clients to confirm participation.
    Starts training if at least one client accepts, regardless of others.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        session_data = federated_manager.get_session(session_id)
        federated_manager.log_event(
            session_data.id,
            f"Starting client confirmation (timeout: {timeout}s)",
            FederatedSessionLogTag.INFO,
        )

        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(5)

        # After timeout, fetch session and decide next step
        federated_session = db.query(FederatedSession).filter_by(id=session_id).first()
        if not federated_session:
            raise Exception(f"Session {session_id} not found in database")
        accepted_clients = session_data.clients

        if accepted_clients:
            federated_session.training_status = 3  # Start training
            federated_manager.log_event(
                session_data.id,
                f"Timeout reached. Starting training with {len(accepted_clients)} clients.",
                FederatedSessionLogTag.TRAINING,
            )
        else:
            federated_session.training_status = -1  # Cancel training
            federated_manager.log_event(
                session_data.id,
                "Timeout reached. No clients accepted. Training canceled.",
                FederatedSessionLogTag.ERROR,
            )
        db.commit()
        return {
            "success": True,
            "training_status": federated_session.training_status,
            "message": "Client confirmation process completed after timeout",
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


async def send_model_configs_and_wait_for_confirmation(
    federated_manager: FederatedLearning,
    session_id: int,
    max_retries: int = 30,
    retry_interval: int = 5,
) -> bool:
    """
    Send model configs to clients and wait for confirmation (status=1).
    Returns True if all clients confirmed, False if timed out.

    Args:
        federated_manager: Federated learning manager
        session_id: ID of the current session
        max_retries: Maximum number of retry attempts
        retry_interval: Seconds between retries
    """
    try:
        with Session(engine) as db:

            session_data = federated_manager.get_session(session_id)
            federated_manager.log_event(
                session_data.id,
                "Starting model config distribution",
                FederatedSessionLogTag.INFO,
            )

            # Get initial interested clients (status=2)
            client_ids = [client.user_id for client in session_data.clients]

            message = {
                "type": MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
                "session_id": session_data.id,
            }
            federated_manager.log_event(
                session_data.id,
                f"Notification:GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS is sent to all users.",
                FederatedSessionLogTag.INFO,
            )
            add_notifications_for(db, message, client_ids)

            # Tracking variables
            attempt = 0
            last_ready_count = 0

            while attempt < max_retries:
                db.expire_all()
                session_data = federated_manager.get_session(session_id)
                current_clients = session_data.clients

                # Get readiness info
                ready_clients = [
                    c
                    for c in current_clients
                    if c.user_id in client_ids and c.status == 1
                ]
                unready_clients = [
                    c
                    for c in current_clients
                    if c.user_id in client_ids and c.status == 0
                ]
                ready_count = len(ready_clients)

                # Log progress if changed
                if ready_count != last_ready_count:
                    federated_manager.log_event(
                        session_data.id,
                        f"Client readiness: {ready_count}/{len(client_ids)} "
                        f"(attempt {attempt + 1}/{max_retries})",
                        FederatedSessionLogTag.INFO,
                    )
                    last_ready_count = ready_count

                # Complete if all ready
                if ready_count == len(client_ids):
                    federated_manager.log_event(
                        session_data.id,
                        "All clients confirmed readiness",
                        FederatedSessionLogTag.SUCCESS,
                    )
                    return True

                # Exponential backoff with jitter
                sleep_time = min(
                    retry_interval * (2 ** (attempt // 3)),
                    60,  # Max 1 minute between retries
                ) * (
                    0.9 + 0.2 * random.random()
                )  # Add jitter

                await asyncio.sleep(sleep_time)
                attempt += 1

            # Timeout handling
            failed_clients = [
                c.user_id
                for c in current_clients
                if c.status != 1 and c.user_id in client_ids
            ]
            federated_manager.log_event(
                session_data.id,
                f"Timeout waiting for clients: {failed_clients}",
                FederatedSessionLogTag.ERROR,
            )
            return False

    except Exception as e:
        with Session(engine) as db:
            session_data = federated_manager.get_session(int(session_id))
            federated_manager.log_event(
                session_data.id,
                f"Error in model config distribution: {str(e)}",
                FederatedSessionLogTag.ERROR,
            )
        return False


async def send_training_signal_and_wait_for_clients_training(
    federated_manager: FederatedLearning, session_id: str
):
    session_data = federated_manager.get_session(int(session_id))
    interested_clients = [client.user_id for client in session_data.clients]
    curr_round = session_data.curr_round

    with Session(engine) as db:
        federated_manager.log_event(
            session_data.id,
            f"Round {curr_round}: START_TRAINING signal sent to {len(interested_clients)} clients",
            FederatedSessionLogTag.TRAINING,
        )
        message = {"type": MessageType.START_TRAINING, "session_id": session_data.id}
        add_notifications_for(db, message, interested_clients)
    federated_manager.log_event(
        session_data.id,
        f"Round {curr_round}: Notification:START_TRAINING is sent to all users.",
        FederatedSessionLogTag.TRAINING,
    )
    federated_manager.log_event(
        session_data.id,
        f"Round {curr_round}: Waiting for clients to complete local training",
        FederatedSessionLogTag.TRAINING,
    )

    # Run the wait_for_all_clients_to_local_training task in the background
    await wait_for_all_clients_to_local_training(federated_manager, session_id)
    federated_manager.log_event(
        session_data.id,
        f"Round {curr_round}:All clients completed local training",
        FederatedSessionLogTag.TRAINING,
    )


async def wait_for_all_clients_to_local_training(
    federated_manager: FederatedLearning,
    session_id: str,
    max_retries: int = 720,
    retry_interval: int = 20,
):
    """
    Wait for all clients to submit their local training parameters.
    Returns True if all clients submitted, False if timed out.

    Args:
        federated_manager: Federated learning manager
        session_id: ID of the current session
        max_retries: Maximum number of retry attempts
        retry_interval: Seconds between retries
    """
    try:
        with Session(engine) as db:
            session_data = federated_manager.get_session(int(session_id))

            # Get the number of interested clients (clients in the session)
            num_interested_clients = len(session_data.clients)
            curr_round = session_data.curr_round
            federated_manager.log_event(
                session_data.id,
                f"Round {curr_round}: Waiting for {num_interested_clients} clients to submit parameters",
                FederatedSessionLogTag.TRAINING,
            )
            # Tracking variables
            attempt = 0
            last_submitted_count = 0
            while attempt < max_retries:
                db.expire_all()
                # Fetch the session again in case it has been updated
                session_data = federated_manager.get_session(int(session_id))

                # Query the submissions of the clients for the current round
                submissions_count = (
                    db.query(FederatedRoundClientSubmission)
                    .filter(
                        FederatedRoundClientSubmission.session_id == session_id,
                        FederatedRoundClientSubmission.round_number == curr_round,
                    )
                    .count()
                )
                federated_manager.log_event(
                    session_data.id,
                    f"Round {curr_round}: Clients who have submitted: {submissions_count}",
                    FederatedSessionLogTag.TRAINING,
                )

                # Log progress if changed
                if submissions_count != last_submitted_count:
                    federated_manager.log_event(
                        session_data.id,
                        f"Round {curr_round}: {submissions_count}/{num_interested_clients} clients submitted (attempt {attempt + 1}/{max_retries})",
                        FederatedSessionLogTag.TRAINING,
                    )
                    last_submitted_count = submissions_count

                # Complete if all clients submitted
                if submissions_count == num_interested_clients:
                    federated_manager.log_event(
                        session_data.id,
                        f"Round {curr_round}: All clients submitted parameters",
                        FederatedSessionLogTag.TRAINING,
                    )
                    return

                # Exponential backoff with jitter
                sleep_time = min(
                    retry_interval * (2 ** (attempt // 3)),
                    60,  # Max 1 minute between retries
                ) * (
                    0.9 + 0.2 * random.random()
                )  # Add jitter

                await asyncio.sleep(sleep_time)
                attempt += 1
            # Timeout handling
            missing_clients = num_interested_clients - last_submitted_count
            federated_manager.log_event(
                session_data.id,
                f"Round {curr_round}: Timeout waiting for {missing_clients} clients to submit parameters",
                FederatedSessionLogTag.TRAINING,
            )
            return
    except Exception as e:
        with Session(engine) as db:
            session_data = federated_manager.get_session(int(session_id))
            federated_manager.log_event(
                session_data.id,
                f"Error waiting for client Local Training submissions: {str(e)}",
                FederatedSessionLogTag.ERROR,
            )
        return
