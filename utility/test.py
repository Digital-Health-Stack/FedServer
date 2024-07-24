import numpy as np
import os
import json
from .ModelBuilder import model_instance_from_config
from .metrics import calculate_metrics


class Test:
    def __init__(self, session_id, session_data):
        self.model = None
        self.session_id = session_id
        self.session_data = session_data
        self.model_config = session_data["federated_info"].dict()  # convert to dict to build the model
        self.metrics = self.model_config["model_info"]["test_metrics"] # metrics to calculate in test
        self.round = 0
        self.test_results = {}
        self.build_model()

    def build_model(self):
        """Build the model for testing"""
        self.model = model_instance_from_config(self.model_config)
        print("Testing model built successfully")

    def start_test(self, updated_weights):
        """Test the model with the updated weights"""

        if self.model is None:
            raise ValueError("Model not built yet...")
        print("Testing model...")

        self.model.update_parameters(updated_weights)
        # read data from file
        try:
            data_dir = os.path.join("utility", "global_test_data")
            X_test = np.load(os.path.join(data_dir, "X_test.npy"))
            Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            return

        Y_pred = self.model.predict(X_test)

        # calculate metrics from calculate_metrics function in metrics.py
        round_results = calculate_metrics(Y_test, Y_pred, self.metrics)
        self.test_results[self.round] = round_results
        self.round += 1

    def save_test_results(self):
        """Save test results to a file"""
        results_dir = "Global_test_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        path = os.path.join(results_dir, f"{self.session_id}_test_results.json")
        with open(path, "w") as f:
            # save results and session data for future reference
            json.dump({"session_data": self.model_config, "test_results": self.test_results}, f)
        print("Results saved successfully view them in formatted way on https://jsonformatter.org/json-viewer")
