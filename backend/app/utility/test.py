import numpy as np
import os
import json
from .model_builder import model_instance_from_config
from .metrics import calculate_metrics
from utility.db import engine
from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession, FederatedTestResults


def save_weights_to_file(weights: dict, filename: str):
    """Save the given weights dictionary to a JSON file."""
    if weights is None:
        weights = {}

    with open(filename, "a") as f:
        json.dump(weights, f, indent=4)
        f.write("\n\n")  # optional: separate rounds visually


class Test:
    def __init__(self, session_id):
        self.model = None
        self.session_id = session_id
        #  Fetch session data and clients within an active session
        with Session(engine) as db:
            session_data = db.query(FederatedSession).filter_by(id=session_id).first()
            if not session_data:
                raise ValueError(f"FederatedSession with ID {session_id} not found")

            # Access the clients relationship within the session
            self.model_config = session_data.federated_info
        print("Model config: ", self.model_config)
        self.metrics = self.model_config["model_info"][
            "test_metrics"
        ]  # metrics to calculate in test
        self.round = session_data.curr_round
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

        # weights_filename = os.path.join("logs", f"weights_round_{self.round}.json")
        # save_weights_to_file(updated_weights, weights_filename)

        self.model.update_parameters(updated_weights)

        # read data from file
        try:
            print("Loading test data...")
            X_test = np.load(os.path.join("data", f"X_{self.session_id}.npy"))
            Y_test = np.load(os.path.join("data", f"Y_{self.session_id}.npy"))
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            return
        # calculate metrics from calculate_metrics function in metrics.py
        # round_results = calculate_metrics(Y_test, Y_pred, self.metrics)
        print("Metrics: ", self.metrics)
        metrics_report = self.model.evaluate(X_test, Y_test, self.metrics)
        with Session(engine) as db:
            test_result = FederatedTestResults(
                session_id=self.session_id,
                round_number=self.round,
                metrics_report=metrics_report,
            )
            db.add(test_result)
            db.commit()
        self.round += 1
        return metrics_report
