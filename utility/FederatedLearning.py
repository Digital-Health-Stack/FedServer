from typing import Dict, List
from schema import FederatedLearningInfo, User
from utility.Server import Server
import numpy as np


class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}
    
    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(self,session_id: str,federated_info: FederatedLearningInfo,clients_data) :
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
        self.federated_sessions[session_id] = {
            "federated_info": federated_info,
            "admin": None,
            "curr_round": 1,
            "max_round": 3,
            "interested_clients": {}, # contains ids of interested_clients
            "global_parameters": [],   # contains global parameters
            "clients_status": {user_id: {"status": 1} for user_id in clients_data},   
            "training_status": 1,            # 1 for server waiting for all clients and 2 for training starts
            "client_parameters": {}
        }
    
    def get_session(self, session_id: str) -> FederatedLearningInfo:
        """
        Retrieves information about a federated learning session.

        Parameters:
        - session_id (str): Unique identifier for the session.

        Returns:
        - FederatedLearningInfo: Information about the federated learning session.
        """
        return self.federated_sessions[session_id]["federated_info"]
    
    def clear_client_parameters(self,session_id: str):
        self.federated_sessions[session_id]['client_parameters'] = {}

    import numpy as np

    def aggregate_weights_fedAvg_Neural(self, session_id: str):
        # Retrieve client parameters
        client_parameters = self.federated_sessions[session_id]['client_parameters']

        # Count the number of clients
        num_interested_clients = len(client_parameters)

        # Initialize a dictionary to hold the aggregated sums of parameters
        aggregated_sums = {}

        def initialize_aggregated_sums(param, aggregated):
            if isinstance(param, list):
                return [initialize_aggregated_sums(sub_param, aggregated) for sub_param in param]
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

        # Iterate over each client's parameters
        for client in client_parameters:
            client_param = client_parameters[client]

            # Initialize aggregated_sums with the same structure as the first client's parameters
            if not aggregated_sums:
                for key in client_param:
                    aggregated_sums[key] = initialize_aggregated_sums(client_param[key], aggregated_sums)

            # Sum the parameters for each key
            for key in client_param:
                aggregated_sums[key] = sum_parameters(aggregated_sums[key], client_param[key])

        # Average the aggregated sums
        for key in aggregated_sums:
            aggregated_sums[key] = average_parameters(aggregated_sums[key], num_interested_clients)

        print("Aggregated Parameters after FedAvg:",
              {k: (type(v), len(v) if isinstance(v, list) else 'N/A') for k, v in aggregated_sums.items()})

        # Save the aggregated parameters back to the session
        self.federated_sessions[session_id]['global_parameters'] = aggregated_sums
        self.clear_client_parameters(session_id)


        """ leave this fucntion as of now..may be of use later"""
        def temp_aggregate_weights_fedAvg_Neural(self, session_id: str):
            # Retrieve client parameters
            client_parameters = self.federated_sessions[session_id]['client_parameters']

            # Count the number of clients
            num_interested_clients = len(client_parameters)

            # Initialize a dictionary to hold the aggregated sums of parameters
            aggregated_sums = {}

            # Iterate over each client's parameters
            for client in client_parameters:
                client_param = client_parameters[client]

                # Initialize aggregated_sums with the same structure as the first client's parameters
                if not aggregated_sums:
                    for key in client_param:
                        for i in range(len(client_param[key])):
                            aggregated_sums[key].append(np.zeros_like(np.array(client_param[key][i])))

                # Sum the parameters for each key
                for key in client_param:
                    for i in range(len(client_param[key])):
                        aggregated_sums[key][i] += np.array(client_param[key][i])

            # Average the aggregated sums
            for key in aggregated_sums:
                for i in range(len(aggregated_sums[key])):
                    aggregated_sums[key][i] /= num_interested_clients
                    aggregated_sums[key][i] = aggregated_sums[key][i].tolist()  # Convert back to list

            print("Aggregated Parameters after FedAvg:",
                  {k: (type(v), len(v) if isinstance(v, list) else 'N/A') for k, v in aggregated_sums.items()})

            # Save the aggregated parameters back to the session
            self.federated_sessions[session_id]['global_parameters'] = aggregated_sums
            self.clear_client_parameters(session_id)
