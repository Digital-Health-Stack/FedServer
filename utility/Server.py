import numpy as np


class Server:
    def __init__(self, globals_parameters, max_round):
        self.globals_parameters = globals_parameters
        self.client_parameters = {}
        self.curr_round = 1
        self.max_round = 5

    def clear_client_parameters(self):
        self.client_parameters = {}

    def aggregate_weights_fedAvg_Neural(self):
        # Initialize a dictionary to hold the aggregated sums of vectors
        # print("Received Parameters : " , type(self.client_parameters[1][1][0]),
        #                                 len(self.client_parameters[1][1]),self.client_parameters[1][2][0][:5])

        # Count the number of clients
        num_clients = len(self.client_parameters)

        client_parameters = self.client_parameters
        for client in client_parameters:
            client_parameters[client] = [np.array(arr) for arr in client_parameters[client]]

        aggregated_sums = []
        for layer in range(len(client_parameters[1])):
            layer_dimension = client_parameters[1][layer].shape

            aggregated_layer = np.zeros(layer_dimension)

            for client in client_parameters:
                aggregated_layer += client_parameters[client][layer]
            aggregated_layer /= num_clients
            aggregated_sums.append(aggregated_layer.tolist())

        print("Aggregate Weights after FedAvg: ", type(aggregated_sums[0][1][0]), len(aggregated_sums[0][1]),
              aggregated_sums[2][0][:4])

        self.globals_parameters = aggregated_sums
        self.clear_client_parameters()
