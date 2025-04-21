
from datetime import datetime
from typing import Dict
from requests import session
from sqlalchemy import Null
from models.FederatedSession import FederatedSession, FederatedSessionClient, FederatedRoundClientSubmission
from utility import FederatedLearning
from models import User
import asyncio
import os
import json
from utility.db import engine
from sqlalchemy.orm import Session
from utility.notification import add_notifications_for, add_notifications_for_user, add_notifications_for_recently_active_users
from utility.SampleSizeEstimation import calculate_required_data_points
from crud.task_crud import (
    get_task_by_id,
)
from constant.message_type import MessageType
import random
from utility.test import Test


def save_weights_to_file(weights: dict, filename: str):
    """Save the given weights dictionary to a JSON file."""
    if weights is None:
        weights = {}

    dirpath = os.path.dirname(filename)
    print(f"path to save weights: {dirpath}")
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    with open(filename, 'a') as f:
        json.dump(weights, f, indent=4)
        
def get_baseline_stats_from_task(db: Session, task_id: int):
    task = get_task_by_id(db, task_id)
    if not task:
        raise ValueError("Task not found!")
    metric_name = task.metric
    if not task.temp_benchmark or metric_name not in task.temp_benchmark:
        raise ValueError(f"No benchmark data found for metric '{metric_name}'")
    benchmark_data = task.temp_benchmark.get(metric_name)
    if not benchmark_data:
        raise ValueError(f"Metric '{metric_name}' not found in temp_benchmark")
    baseline_mean = benchmark_data.get("std_mean")
    baseline_std = benchmark_data.get("std_dev")
    if baseline_mean is None or baseline_std is None:
        raise ValueError(f"Incomplete benchmark data for '{metric_name}'")

    return baseline_mean, baseline_std
        


def fetch_benchmark_and_calculate_price(session_data: FederatedSession, db: Session)->float :
    
    task_id = session_data.federated_info["dataset_info"].get("task_id")
    baseline_mean, baseline_std = get_baseline_stats_from_task(db, task_id)
    
    try:
        new_mean = float(session_data.federated_info["expected_results"].get("std_mean"))
        new_std = float(session_data.federated_info["expected_results"].get("std_deviation"))
    except (TypeError, ValueError) as e:
        raise ValueError("Expected results must contain valid float values for std_mean and std_deviation.") from e

    if new_mean is None or new_std is None:
        raise ValueError("New model metrics (std_mean and std_deviation) are missing in session data.")
    
    # Extract num_predictors strictly from input_shape
    model_config = session_data.federated_info

    if not model_config:
        raise ValueError("Model Config is missing in model_info.")

    # Calculate the required data points (price)
    price = calculate_required_data_points(
        model_config, baseline_mean, baseline_std, new_mean, new_std
    )
    return price
    
async def start_federated_learning(federated_manager: FederatedLearning, user: User, session_data: FederatedSession, db: Session):
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
    federated_manager.log_event(session_data.id, "Starting federated learning process")
    
    federated_manager.log_event(session_data.id, "Fetching benchmark stats and calculating training price")
    required_data_points = fetch_benchmark_and_calculate_price(session_data, db)
    federated_manager.log_event(session_data.id, f"Calculated training price as {required_data_points} data points")
    
    
    # Store the calculated price in the session
    federated_manager.log_event(session_data.id, f"Storing calculated price in session {session_data.id}")
    federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
    if federated_session:    
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
        federated_manager.log_event(session_data.id, "Price successfully stored in session")
    else:
        error_msg = f"FederatedSession with ID {session_data.id} not found."
        federated_manager.log_event(session_data.id, f"{error_msg}")
        return
    
    # Send the price to the client and wait for approval
    federated_manager.log_event(session_data.id,"Waiting for client price confirmation")
    approved = await wait_for_price_confirmation(federated_manager, session_data.id, db)

    if not approved:
        federated_session = db.query(FederatedSession).filter_by(id = session_data.id).first()
        federated_session.training_status = -1
        db.commit()
        federated_manager.log_event(session_data.id, f"Client {user.id} declined the price. Training aborted.")
        return

    federated_manager.log_event(session_data.id, f"Client {user.id} accepted the price. Training starts.")
    
    message = {
        'type': "new-session",
        'message': "New Federated Session Avaliable!",
        'session_id': session_data.id
    }
    
    # Make alert in Client side
    federated_manager.log_event(session_data.id, "Sending notifications to active users")
    add_notifications_for_recently_active_users(db=db, message=message, valid_until=session_data.wait_till, excluded_users=[user])
    federated_manager.log_event(session_data.id, f"Notification:new-session is sent to all users.")
    
    # Wait for client confirmation of interest
    await wait_for_client_confirmation(federated_manager, session_data.id, db)
    federated_manager.log_event(session_data.id, "Client confirmation process completed.")
    
    # Send Model Configurations to interested clients and wait for their confirmation
    success = await send_model_configs_and_wait_for_confirmation(
        federated_manager, 
        session_data.id
    )
    if success:
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
            federated_session.training_status = 4
            db.commit()
        federated_manager.log_event(session_data.id, "All interested clients confirmed model configuration")
    else:
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
            federated_session.training_status = -1
            db.commit()
        federated_manager.log_event(
            session_data.id, 
            "Timeout: Not all clients confirmed model configuration"
        )
        return
    
    #############################################
    # code used to get instance of testing unit
    # Here Input has to be taken in future for the metrics
    test = Test(session_data.id)
    federated_manager.log_event(session_data.id, f"Initialized test unit.")

    # Download data from Hadoop
    federated_info = session_data.federated_info
    dataset_info = federated_info.get("dataset_info")
    client_filename = dataset_info.get("client_filename")
    output_columns = dataset_info.get("output_columns")
    process_parquet_and_save_xy(client_filename, session_id, output_columns)
    
    # Start Training
    federated_manager.log_event(session_data.id, f"Starting training with {session_data.max_round} rounds")
    for i in range(1, session_data.max_round + 1):
        session_data = federated_manager.get_session(session_data.id)
        federated_manager.log_event(session_data.id, f"Starting round {i}.")
        with Session(engine) as db:
            # Update the training status of the session
            federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
            federated_session.curr_round = i
            db.commit()
        federated_manager.log_event(session_data.id, f"Round {i} marked in database")
        
        await send_training_signal_and_wait_for_clients_training(federated_manager, session_data.id)
        # Aggregate
        federated_manager.log_event(session_data.id, f"Performing aggregation.")
        before_global_weights = federated_manager.get_latest_global_weights(session_data.id)
        save_weights_to_file(before_global_weights, "logs/before_weights.json")
        
        federated_manager.aggregate_weights_fedAvg_Neural(session_data.id, i)
        
        federated_manager.log_event(session_data.id, f"Aggregation is done")
        after_global_weights = federated_manager.get_latest_global_weights(session_data.id)
        save_weights_to_file(after_global_weights, "logs/after_weights.json")
        
        with Session(engine) as db:
            federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
            if not federated_session:
                error_msg = f"FederatedSession not found during round {i}."
                federated_manager.log_event(session_data.id, error_msg)
                raise ValueError(error_msg)

            ################ Testing start
            results = test.start_test(federated_manager.get_latest_global_weights(session_data.id))
            federated_manager.log_event(session_data.id, f"Global test results: {results}")
            
            # Reset client_parameters to an empty JSON object
            
            federated_manager.log_event(session_data.id, f"Client parameters reset after Round {i}.")
            db.commit()  # Save the reset to the database

    with Session(engine) as db:
        # Update the training status of the session
        federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
        federated_session.training_status = 5
        db.commit()
    federated_manager.log_event(session_data.id, f"Training completed. Test results saved.")


async def wait_for_price_confirmation(federated_manager: FederatedLearning, session_id: str, db: Session, timeout: int = 3000):
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
    session_data = federated_manager.get_session(session_id)
    federated_manager.log_event(session_data.id, f"Starting price confirmation wait (timeout: {timeout}s)")
    
    while True:
        session_data = federated_manager.get_session(session_id)

        if session_data.training_status == 2:  # Status 2 means price was accepted
            federated_manager.log_event(session_data.id, f"Client accepted the price for session {session_id}")
            return True

        # Check for timeout
        if asyncio.get_event_loop().time() - start_time > timeout:
            federated_manager.log_event(session_data.id, f"Timeout: Client did not confirm price within {timeout}s")
            return False

        federated_manager.log_event(session_data.id, "Waiting for client price confirmation.")
        await asyncio.sleep(5)  # Non-blocking sleep

async def wait_for_client_confirmation(
    federated_manager: FederatedLearning,
    session_id: int,
    db: Session,
    timeout: int = 30
):
    """
    Waits for a fixed timeout period for clients to confirm participation.
    Starts training if at least one client accepts, regardless of others.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        session_data = federated_manager.get_session(session_id)
        federated_manager.log_event(session_data.id, f"Starting client confirmation (timeout: {timeout}s)")

        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(5)

        # After timeout, fetch session and decide next step
        federated_session = db.query(FederatedSession).filter_by(id=session_id).first()
        if not federated_session:
            raise Exception(f"Session {session_id} not found in database")
        accepted_clients = session_data.clients

        if accepted_clients:
            federated_session.training_status = 3  # Start training
            federated_manager.log_event(session_data.id, f"Timeout reached. Starting training with {len(accepted_clients)} clients.")
        else:
            federated_session.training_status = -1  # Cancel training
            federated_manager.log_event(session_data.id, "Timeout reached. No clients accepted. Training canceled.")
        db.commit()
        return {
            'success': True,
            'training_status': federated_session.training_status,
            'message': 'Client confirmation process completed after timeout'
        }

    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }

# async def send_model_configs_and_wait_for_confirmation(federated_manager: FederatedLearning, session_id: int):
#     session_data = federated_manager.get_session(session_id)
#     federated_manager.log_event(session_data.id, "Preparing to send model configurations")
    
#     interested_clients = [client.user_id for client in session_data.clients if client.status == 2]
    
#     model_config = session_data.federated_info
    
#     message = {
#         "type": MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
#         "data": model_config,
#         "session_id": session_data.id
#     }

#     with Session(engine) as db:
#         federated_manager.log_event(session_data.id, f"Sending model configs to {len(interested_clients)} clients")
#         add_notifications_for(db, message, interested_clients)

#     # Wait for all clients to confirm they have started their background process
#     await wait_for_clients_initiate_model(session_data)
#     federated_manager.log_event(session_data.id, "All clients confirmed model config receipt")
    

# Need to rethink
# async def wait_for_clients_initiate_model(session_data: FederatedSession):
#     # Implement the logic to wait for all clients to confirm that they have started background process
#     with Session(engine) as db:
#         federated_manager.log_event(session_data.id, "Checking for client local model IDs")
#         while True:
#             # Expire all objects in the session to force reloading
#             db.expire_all()
            
#             interested_clients = db.query(FederatedSessionClient).filter_by(session_id = session_data.id).all()
#             all_clients_ready = True
#             for client in interested_clients:
#                 if client.local_model_id == None:
#                     all_clients_ready = False
#                     break

#             if all_clients_ready:
#                 federated_manager.log_event(session_data.id, "All clients provided local model IDs")
#                 break
#             else:
#                 federated_manager.log_event(session_data.id, "Waiting for clients to provide local model IDs...")
#                 await asyncio.sleep(5)

async def send_model_configs_and_wait_for_confirmation(
    federated_manager: FederatedLearning, 
    session_id: int,
    max_retries: int = 30,
    retry_interval: int = 5
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
            federated_manager.log_event(session_data.id, "Starting model config distribution")
            
            # Get initial interested clients (status=2)
            client_ids = [client.user_id for client in session_data.clients]

            message = {
                "type": MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
                "session_id": session_data.id
            }
            

            # Tracking variables
            attempt = 0
            last_ready_count = 0
            
            while attempt < max_retries:
                db.expire_all()
                session_data = federated_manager.get_session(session_id)
                current_clients = session_data.clients
                
                # Get readiness info
                ready_clients = [c for c in current_clients if c.user_id in client_ids and c.status == 1]
                unready_clients = [c for c in current_clients if c.user_id in client_ids and c.status == 0]
                ready_count = len(ready_clients)
                
                # Send notifications only to unready clients
                unready_ids = [c.user_id for c in unready_clients]
                add_notifications_for(db, message, unready_ids)
                federated_manager.log_event(session_data.id, f"Notification:GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS is sent to all users {attempt}th time.")
                # Log progress if changed
                if ready_count != last_ready_count:
                    federated_manager.log_event(
                        session_data.id,
                        f"Client readiness: {ready_count}/{len(client_ids)} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    last_ready_count = ready_count
                
                # Complete if all ready
                if ready_count == len(client_ids):
                    federated_manager.log_event(session_data.id, "All clients confirmed readiness")
                    return True
                
                # Exponential backoff with jitter
                sleep_time = min(
                    retry_interval * (2 ** (attempt // 3)),
                    60  # Max 1 minute between retries
                ) * (0.9 + 0.2 * random.random())  # Add jitter
                
                await asyncio.sleep(sleep_time)
                attempt += 1
            
            # Timeout handling
            failed_clients = [
                c.user_id for c in current_clients 
                if c.status != 1 and c.user_id in client_ids
            ]
            federated_manager.log_event(
                session_data.id,
                f"Timeout waiting for clients: {failed_clients}",
                is_error=True
            )
            return False
            
    except Exception as e:
        with Session(engine) as db:
            session_data = federated_manager.get_session(session_id)
            federated_manager.log_event(
                session_data.id,
                f"Error in model config distribution: {str(e)}"
            )
        return False


async def send_training_signal_and_wait_for_clients_training(federated_manager: FederatedLearning, session_id: str):
    session_data = federated_manager.get_session(session_id)
    interested_clients = [client.user_id for client in session_data.clients]
    curr_round = session_data.curr_round
    
    with Session(engine) as db:
        federated_manager.log_event(session_data.id, f"Round {curr_round}: START_TRAINING signal sent to {len(interested_clients)} clients")
        message = {
            "type": MessageType.START_TRAINING,
            "session_id": session_data.id
        }
        add_notifications_for(db, message, interested_clients)
        federated_manager.log_event(session_data.id, f"Round {curr_round}: Notification:START_TRAINING is sent to all users.")
        federated_manager.log_event(session_data.id, f"Round {curr_round}: Waiting for clients to complete local training")
    
        # Run the wait_for_all_clients_to_local_training task in the background
        await wait_for_all_clients_to_local_training(federated_manager, session_id)
        federated_manager.log_event(session_data.id, f"Round {curr_round}:All clients completed local training")

# async def wait_for_all_clients_to_local_training(federated_manager: FederatedLearning, session_id: str):
#     with Session(engine) as db:
#         session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
        
#         if not session:
#             federated_manager.log_event(session_data.id, f"Session {session_data.id} not found")
#             raise ValueError(f"FederatedSession with ID {session.id} not found.")
        
#         num_interested_clients = len(session.clients)
#         federated_manager.log_event(session_data.id, f"Total interested clients: {num_interested_clients}")
        
#         while True:
#             # Expire all objects in the session to force reloading
#             db.expire_all()
#             session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
#             if not session:
#                 federated_manager.log_event(session_data.id, f"Session {session_data.id} not found (possibly deleted)")
#                 raise ValueError(f"FederatedSession with ID {session_data.id} not found (possibly deleted).")

            
#             # Deserialize client_parameters from JSON to Python dict
#             client_parameters = json.loads(session.client_parameters) if session.client_parameters else {}


#             # Count clients with local model parameters submitted
#             num_clients_with_local_models = len([
#                 client_id for client_id, params in client_parameters.items()
#             ])
            
            
#             federated_manager.log_event(session_data.id, f"Training progress: {num_clients_with_local_models}/{num_interested_clients} clients ready")
            
#             # Check if all interested clients have submitted their parameters
#             if num_clients_with_local_models >= num_interested_clients:
#                 federated_manager.log_event(session_data.id, "All clients completed local training")
#                 break
#             else:
#                 federated_manager.log_event(session_data.id, f"Waiting for {num_interested_clients - num_clients_with_local_models} more clients")
#                 await asyncio.sleep(5)


async def wait_for_all_clients_to_local_training(federated_manager: FederatedLearning, session_id: str,max_retries: int = 30, retry_interval: int = 5):
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
            session_data = federated_manager.get_session(session_id)
            
            # Get the number of interested clients (clients in the session)
            num_interested_clients = len(session_data.clients)
            curr_round = session_data.curr_round
            federated_manager.log_event(session_data.id, f"Round {curr_round}: Waiting for {num_interested_clients} clients to submit parameters")
            # Tracking variables
            attempt = 0
            last_submitted_count = 0
            while attempt < max_retries:
                db.expire_all()
                # Fetch the session again in case it has been updated
                session_data = federated_manager.get_session(session_id)
                
                # Query the submissions of the clients for the current round
                submissions_count = db.query(FederatedRoundClientSubmission).filter(
                    FederatedRoundClientSubmission.session_id == session_id,
                    FederatedRoundClientSubmission.round_number == curr_round
                ).count()
                federated_manager.log_event(session_data.id, f"Round {curr_round}: Clients who have submitted: {submissions_count}")
                
                # Log progress if changed
                if submissions_count != last_submitted_count:
                    federated_manager.log_event(session_data.id,f"Round {curr_round}: {submissions_count}/{num_interested_clients} clients submitted (attempt {attempt + 1}/{max_retries})")
                    last_submitted_count = submissions_count
                    
                # Complete if all clients submitted
                if submissions_count == num_interested_clients:
                    federated_manager.log_event(session_data.id, f"Round {curr_round}: All clients submitted parameters")
                    return True
                
                # Exponential backoff with jitter
                sleep_time = min(
                    retry_interval * (2 ** (attempt // 3)),
                    60  # Max 1 minute between retries
                ) * (0.9 + 0.2 * random.random())  # Add jitter
                    
                await asyncio.sleep(sleep_time)
                attempt += 1
            # Timeout handling
            missing_clients = num_interested_clients - last_submitted_count
            federated_manager.log_event(session_data.id, f"Round {curr_round}: Timeout waiting for {missing_clients} clients to submit parameters")
            return False
    except Exception as e:
        with Session(engine) as db:
            session_data = federated_manager.get_session(session_id)
            federated_manager.log_event(
                session_data.id,
                f"Error waiting for client Local Training submissions: {str(e)}"
            )
        return False
            
            
            
            
            
            
            
        
        
        
        
        
