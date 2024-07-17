import numpy as np
from schema import FederatedLearningInfo


class FederatedLearning:
    def __init__(self):
        self.federated_sessions = {}

    # Every session has a session_id also in future we can add a token and id
    def create_federated_session(self, session_id: str, federated_info: FederatedLearningInfo, clients_data):
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
            "max_round": 5,
            "interested_clients": {},  # contains ids of interested_clients
            "global_parameters": [],  # Contains user id of interested students
            "clients_status": {user_id: {"status": 1} for user_id in clients_data},
            "training_status": 1,  # 1 for server waiting for all clients and 2 for training starts
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

    def clear_client_parameters(self, session_id: str):
        self.federated_sessions[session_id]['client_parameters'] = {}

    def aggregate_weights_fedAvg_Neural(self, session_id: str):
        # Initialize a dictionary to hold the aggregated sums of vectors
        # print("Received Parameters : " , type(self.client_parameters[1][1][0]),
        #                                 len(self.client_parameters[1][1]),self.client_parameters[1][2][0][:5])

        # Count the number of clients
        num_interested_clients = len(self.federated_sessions[session_id]['client_parameters'])

        client_parameters = self.federated_sessions[session_id]['client_parameters']
        for client in client_parameters:
            client_parameters[client] = [np.array(arr) for arr in client_parameters[client]]

        aggregated_sums = []
        for layer in range(len(client_parameters[1])):
            layer_dimension = client_parameters[1][layer].shape

            aggregated_layer = np.zeros(layer_dimension)

            for client in client_parameters:
                aggregated_layer += client_parameters[client][layer]
            aggregated_layer /= num_interested_clients
            aggregated_sums.append(aggregated_layer.tolist())

        print("Shape of Aggregate Weights after FedAvg: ", np.array(aggregated_sums).shape)

        self.federated_sessions[session_id]['globals_parameters'] = aggregated_sums
        self.clear_client_parameters(session_id)
