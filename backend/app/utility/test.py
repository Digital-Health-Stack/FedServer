import numpy as np
import os
import json
from .ModelBuilder import model_instance_from_config
from .metrics import calculate_metrics
from utility.db import engine
from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession

class Test:
    def __init__(self, session_id, session_data):
        self.model = None
        self.session_id = session_id
         # Fetch session data and clients within an active session
        with Session(engine) as db:
            session_data = db.query(FederatedSession).filter_by(id=session_id).first()
            if not session_data:
                raise ValueError(f"FederatedSession with ID {session_id} not found")
            
            # Access the clients relationship within the session
            self.session_data = session_data
            self.num_clients = len(session_data.clients)  # Length of interested clients
        self.model_config = session_data.federated_info 
        self.metrics = self.model_config['model_info']['test_metrics'] # metrics to calculate in test
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
        # setting result with key "round 1", "round 2", etc
        self.test_results[f"round {self.round}"] = round_results
        self.round += 1
        return round_results

    def save_test_results(self):
        """Save test results to a file"""
        results_dir = "Global_test_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        path = os.path.join(results_dir, f"{self.session_id}_test_results.json")
        with open(path, "w") as f:
            # Save results and session data with pretty printing
            json.dump(
                {"session_data": self.model_config, "num_clients": self.num_clients, "test_results": self.test_results},
                f,
                indent=4,  # Add indentation for pretty printing
                separators=(',', ': ')  # Add a space after colon for readability
            )
        print("Results saved successfully. View them in a formatted way on https://jsonformatter.org/json-viewer")
