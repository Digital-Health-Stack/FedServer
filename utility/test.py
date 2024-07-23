import numpy as np
import os
from .ModelBuilder import model_instance_from_config
import json


class Test():
    def __init__(self, session_id, session_data, metrics=None):
        self.model = None
        self.session_id = session_id
        self.metrics = metrics
        self.session_data = session_data
        self.model_config = session_data["federated_info"].dict()  #dict to build the model
        self.round = 0
        self.test_results = {}
        self.build_model()

    def build_model(self):
        """ Build the model for testing """
        self.model = model_instance_from_config(self.model_config)
        print("Testing model built successfully")

    def start_test(self, updated_weights):
        """ Test the model with the updated weights """

        if self.model is None or self.metrics is None:
            raise ValueError("Model not built yet or None metrics passed")
        print("Testing model...")

        self.model.update_parameters(updated_weights)
        # read data from file
        try:
            X_test = np.load(os.path.join("global_test_data", "X_test.npy"))
            Y_test = np.load(os.path.join("global_test_data", "Y_test.npy"))
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            return

        Y_pred = self.model.predict(X_test)

        round_results = {}
        for metric in self.metrics:
            if metric == "accuracy":
                accuracy = np.mean(Y_pred == Y_test)
                round_results["accuracy"] = accuracy
            elif metric == "mse":
                mse = np.mean((Y_pred - Y_test) ** 2)
                round_results["mse"] = mse
            elif metric == "mae":
                mae = np.mean(np.abs(Y_pred - Y_test))
                round_results["mae"] = mae
            else:
                print(f"Unknown metric: {metric}")

        self.test_results[self.round] = round_results
        self.round += 1

    def save_test_results(self):
        """ Save test results to a file """

        path = os.path.join("Global_test_results", f"{self.session_id}_test_results.json")
        with open(path, "w") as f:
            #  save results and session data for future reference
            json.dump({"session_data": self.model_config, "test_results": self.test_results}, f)
        print("Results saved successfully")


