
from datetime import datetime
from typing import Dict
from requests import session
from sqlalchemy import Null
from models.FederatedSession import FederatedSession, FederatedSessionClient
from utility import FederatedLearning
from models import User
import asyncio
import json
from utility.db import engine
from sqlalchemy.orm import Session
from models.Notification import Notification
from utility.test import Test
from models.Benchmark import Benchmark
from utility.notification import add_notifications_for, add_notifications_for_user, add_notifications_for_recently_active_users
from utility.SampleSizeEstimation import calculate_required_data_points
from crud.task_crud import (
    get_task_by_id,
)


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
    input_shape = session_data.federated_info["model_info"].get("input_shape")

    if not input_shape:
        raise ValueError("Input shape is missing in model_info.")

    try:
        shape_tuple = eval(input_shape)  # Convert string "(128,128,1)" → tuple (128,128,1)
        num_predictors = 1
        for dim in shape_tuple:
            num_predictors *= dim  # Compute total number of input features
    except Exception:
        raise ValueError(f"Invalid input_shape format: {input_shape}")
    # Calculate the required data points (price)
    price = calculate_required_data_points(
        baseline_mean, baseline_std, new_mean, new_std, num_predictors
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
    session_data.log_event(db, "Starting federated learning process")
    
    session_data.log_event(db, "Fetching benchmark stats and calculating training price")
    required_data_points = fetch_benchmark_and_calculate_price(session_data, db)
    session_data.log_event(db, f"Calculated training price as {required_data_points} data points")
    
    
    # Store the calculated price in the session
    session_data.log_event(db, f"Storing calculated price in session {session_data.id}")
    federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
    if federated_session:    
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
        session_data.log_event(db, "Price successfully stored in session")
    else:
        error_msg = f"FederatedSession with ID {session_data.id} not found."
        session_data.log_event(db, f"Session {session_data.id}: {error_msg}")
        return
    
    # Send the price to the client and wait for approval
    session_data.log_event(db, "Waiting for client price confirmation")
    approved = await wait_for_price_confirmation(federated_manager, session_data.id, db)

    if not approved:
        session_data.log_event(db, f"Client {user.id} declined the price. Training aborted.")
        return

    session_data.log_event(db, f"Client {user.id} accepted the price. Training starts.")
    
    # message = {
    #     'type': "new-session",
    #     'message': "New Federated Session Avaliable!",
    #     'session_id': session_data.id
    # }
    
    # # Make alert in Client side
    # session_data.log_event(db, "Sending notifications to active users")
    # add_notifications_for_recently_active_users(db=db, message=message, valid_until=session_data.wait_till, excluded_users=[user])
    # session_data.log_event(db, f"Notification:new-session is sent to all users.")
    
    # # Wait for client confirmation of interest
    # await wait_for_client_confirmation(federated_manager, session_data.id, db)

    # # Send Model Configurations to interested clients and wait for their confirmation
    # await send_model_configs_and_wait_for_confirmation(federated_manager, session_data.id)
    
    # #############################################
    # # code used to get instance of testing unit
    # # Here Input has to be taken in future for the metrics
    # test = Test(session_data.id, session_data)
    # session_data.log_event(db, f"Initialized test unit.")

    # # Start Training
    # session_data.log_event(db, f"Starting training with {session_data.max_round} rounds")
    # for i in range(1, session_data.max_round + 1):
    #     session_data.log_event(db, f"Starting round {i}.")
    #     federated_session = db.query(FederatedSession).filter_by(id = session_data.id).first()
    #     federated_session.curr_round = i
    #     db.commit()
    #     session_data.log_event(db, f"Round {i} marked in database")
        
    #     await send_training_signal_and_wait_for_clients_training(federated_manager, session_data.id)
    #     # Aggregate
    #     session_data.log_event(db, f"Performing aggregation for round {i}.")
    #     federated_manager.aggregate_weights_fedAvg_Neural(session_data.id)
    #     with Session(engine) as db:
    #         federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
    #         if not federated_session:
    #             error_msg = f"FederatedSession not found during round {i}."
    #             session_data.log_event(db, error_msg)
    #             raise ValueError(error_msg)
            
    #         ################# Testing start
    #         results = test.start_test(json.loads(federated_session.global_parameters))
    #         session_data.log_event(db, f"Global test results: {results}")
            
    #         # Reset client_parameters to an empty JSON object
    #         federated_session.client_parameters = "{}"  # Empty JSON object
    #         db.commit()  # Save the reset to the database
    #         session_data.log_event(db, f"Client parameters reset after Round {i}.")

    #     # Save test results for future reference
    #     """ Yashvir: here you can delete the data of this session from the federated_sessions dictionary after saving the results 
    #         , saved results contains session_data and test_results across all rounds
    #     """
    # test.save_test_results()
    session_data.log_event(db, f"Training completed. Test results saved.")


async def wait_for_price_confirmation(federated_manager: FederatedLearning, session_id: str, db: Session, timeout: int = 300):
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
    session_data.log_event(db, f"Starting price confirmation wait (timeout: {timeout}s)")
    
    while True:
        session_data = federated_manager.get_session(session_id)

        if session_data.training_status == 2:  # Status 2 means price was accepted
            session_data.log_event(db, f"Client accepted the price for session {session_id}")
            return True

        # Check for timeout
        if asyncio.get_event_loop().time() - start_time > timeout:
            session_data.log_event(db, f"Timeout: Client did not confirm price within {timeout}s")
            return False

        session_data.log_event(db, "Waiting for client price confirmation.")
        await asyncio.sleep(5)  # Non-blocking sleep

# async def wait_for_client_confirmation(federated_manager: FederatedLearning, session_id: int):
#     all_ready_for_training = False

#     while not all_ready_for_training:
#         session_data = federated_manager.get_session(session_id)
#         await asyncio.sleep(5)
#         now = datetime.now()
#         all_ready_for_training = all(client.status != 1 for client in session_data.clients) and session_data.wait_till < now
        
#         print("Waiting for client confirmations....Stage 1")

#     # Update Training Status to 3 in training
#     print("All Clients have taken their decision.")

async def wait_for_client_confirmation(
    federated_manager: FederatedLearning,
    session_id: int,
    db: Session,
    timeout: int = 300
):
    """
    Waits for a fixed timeout period for clients to confirm participation.
    Starts training if at least one client accepts, regardless of others.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        session_data = federated_manager.get_session(session_id)
        session_data.log_event(db, f"Starting client confirmation (timeout: {timeout}s)")

        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(5)

        # After timeout, fetch session and decide next step
        federated_session = db.query(FederatedSession).filter_by(id=session_id).first()
        if not federated_session:
            raise Exception(f"Session {session_id} not found in database")

        accepted_clients = [c for c in session_data.clients if c.status == 2]

        if accepted_clients:
            federated_session.training_status = 3  # Start training
            session_data.log_event(db, f"Timeout reached. Starting training with {len(accepted_clients)} clients.")
        else:
            federated_session.training_status = -1  # Cancel training
            session_data.log_event(db, "Timeout reached. No clients accepted. Training canceled.")

        db.commit()
        session_data.log_event(db, "Client confirmation process completed.")

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


class MessageType:
    GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS = "get_model_parameters_start_background_process"
    START_TRAINING = "start_training"

async def send_model_configs_and_wait_for_confirmation(federated_manager: FederatedLearning, session_id: int):

    session_data = federated_manager.get_session(session_id)
    session_data.log_event(db, "Preparing to send model configurations")
    
    interested_clients = [client.user_id for client in session_data.clients if client.status == 2]
    
    model_config = session_data.federated_info
    
    message = {
        "type": MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
        "data": model_config,
        "session_id": session_data.id
    }

    with Session(engine) as db:
        session_data.log_event(db, f"Sending model configs to {len(interested_clients)} clients")
        add_notifications_for(db, message, interested_clients)

    # Wait for all clients to confirm they have started their background process
    await wait_for_all_clients_to_stage_four(session_data)
    session_data.log_event(db, "All clients confirmed model config receipt")

# async def send_message_with_type(client: FederatedSessionClient, message_type: str, data: dict, session_data: FederatedSession):
#     message = {
#         "type": message_type,
#         "data": data,
#         "session_id": session_data.id
#     }

#     json_message = json.dumps(message)

#     print("json model sent before the training signal: ", json_message)
#     print(f"client id {client.id}")
    

# Need to rethink
async def wait_for_all_clients_to_stage_four(session_data: FederatedSession):
    # Implement the logic to wait for all clients to confirm that they have started background process
    with Session(engine) as db:
        session_data.log_event(db, "Checking for client local model IDs")
        while True:
            # Expire all objects in the session to force reloading
            db.expire_all()
            
            interested_clients = db.query(FederatedSessionClient).filter_by(session_id = session_data.id).all()
            all_clients_ready = True
            for client in interested_clients:
                if client.local_model_id == None:
                    all_clients_ready = False
                    break

            if all_clients_ready:
                session_data.log_event(db, "All clients provided local model IDs")
                break
            else:
                session_data.log_event(db, "Waiting for clients to provide local model IDs...")
                await asyncio.sleep(5)


async def send_training_signal_and_wait_for_clients_training(federated_manager: FederatedLearning, session_id: str):
    session_data = federated_manager.get_session(session_id)
    session_data.log_event(db, "Preparing to send training signals")
    interested_clients = [client.user_id for client in session_data.clients if client.status == 4]
    
    model_config = session_data.federated_info
    with Session(engine) as db:
        for client in session_data.clients:
            if client.user_id in interested_clients:
                local_model_id = client.local_model_id
                
                message = {
                    "type": MessageType.START_TRAINING,
                    "data": {
                        "model_config": model_config,
                        "local_model_id": local_model_id  # Wrap both in the `data` key
                    },
                    "session_id": session_data.id,
                }
                session_data.log_event(db, f"Sending training signal to client {client.user_id}")
                add_notifications_for_user(db, client.user_id, message)
        
    session_data.log_event(db, "Waiting for clients to complete local training")
    # Run the wait_for_all_clients_to_local_training task in the background
    await wait_for_all_clients_to_local_training(session_data)
    session_data.log_event(db, "All clients completed local training")

async def wait_for_all_clients_to_local_training(session_data: FederatedSession):
    # Implement the logic to wait for all clients to confirm that they have started background process
    # session_data = federated_manager.federated_sessions[session_id]
    # interested_clients = [client for client in session_data.clients if client.status == 2]
    with Session(engine) as db:
        session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
        
        if not session:
            session_data.log_event(db, f"Session {session_data.id} not found")
            raise ValueError(f"FederatedSession with ID {session.id} not found.")
        
        num_interested_clients = len(session.clients)
        session_data.log_event(db, f"Total interested clients: {num_interested_clients}")
        
        while True:
            # Expire all objects in the session to force reloading
            db.expire_all()
            session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
            if not session:
                session_data.log_event(db, f"Session {session_data.id} not found (possibly deleted)")
                raise ValueError(f"FederatedSession with ID {session_data.id} not found (possibly deleted).")

            
            # Deserialize client_parameters from JSON to Python dict
            client_parameters = json.loads(session.client_parameters) if session.client_parameters else {}


            # Count clients with local model parameters submitted
            num_clients_with_local_models = len([
                client_id for client_id, params in client_parameters.items()
            ])
            
            
            session_data.log_event(db, f"Training progress: {num_clients_with_local_models}/{num_interested_clients} clients ready")
            
            # Check if all interested clients have submitted their parameters
            if num_clients_with_local_models >= num_interested_clients:
                session_data.log_event(db, "All clients completed local training")
                break
            else:
                session_data.log_event(db, f"Waiting for {num_interested_clients - num_clients_with_local_models} more clients")
                await asyncio.sleep(5)

