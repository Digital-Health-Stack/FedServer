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
        # print("Model config: ", self.model_config)
        self.metrics = self.model_config["model_info"][
            "test_metrics"
        ]  # metrics to calculate in test
        self.round = session_data.curr_round
        self.build_model()

    def build_model(self):
        """Build the model for testing"""
        self.model = model_instance_from_config(self.model_config)
        print("Testing model built successfully")

    def temporary_evaluate_function(self):
        """
        Temporary function that simulates improving metrics over federated learning rounds.
        Returns better metrics for each subsequent round.
        """
        import random

        # Set seed based on session_id and round for consistent but improving results
        random.seed(hash(f"{self.session_id}_{self.round}") % 1000000)

        metrics_report = {}

        # Base improvement factor - metrics get better with more rounds
        improvement_factor = min(
            0.95 + (self.round * 0.02), 0.94
        )  # Caps at 99% improvement
        base_performance = 0.3 + (
            self.round * 0.05
        )  # Starting performance improves with rounds

        for metric in self.metrics:
            if metric in ["mse", "mae", "rmse", "msle", "mape", "log_loss"]:
                # For error metrics, lower is better - start high and decrease
                if metric == "mse":
                    base_value = 2.0 - (self.round * 0.15)
                    noise = random.uniform(-0.05, 0.02)  # Small random variation
                    value = max(0.001, base_value + noise)
                elif metric == "mae":
                    base_value = 1.5 - (self.round * 0.12)
                    noise = random.uniform(-0.04, 0.02)
                    value = max(0.001, base_value + noise)
                elif metric == "rmse":
                    base_value = 1.4 - (self.round * 0.11)
                    noise = random.uniform(-0.04, 0.02)
                    value = max(0.001, base_value + noise)
                elif metric == "msle":
                    base_value = 0.8 - (self.round * 0.08)
                    noise = random.uniform(-0.03, 0.01)
                    value = max(0.001, base_value + noise)
                elif metric == "mape":
                    base_value = 25.0 - (self.round * 2.0)
                    noise = random.uniform(-1.0, 0.5)
                    value = max(0.1, base_value + noise)
                elif metric == "log_loss":
                    base_value = 1.2 - (self.round * 0.09)
                    noise = random.uniform(-0.03, 0.01)
                    value = max(0.001, base_value + noise)

                metrics_report[metric] = round(value, 3)

            elif metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "r2_score",
            ]:
                # For performance metrics, higher is better - start low and increase
                if metric == "accuracy":
                    base_value = 0.4 + (self.round * 0.08)
                    noise = random.uniform(-0.02, 0.05)
                    value = min(0.99, base_value + noise)
                elif metric == "precision":
                    base_value = 0.35 + (self.round * 0.07)
                    noise = random.uniform(-0.02, 0.04)
                    value = min(0.98, base_value + noise)
                elif metric == "recall":
                    base_value = 0.38 + (self.round * 0.075)
                    noise = random.uniform(-0.02, 0.04)
                    value = min(0.97, base_value + noise)
                elif metric == "f1_score":
                    base_value = 0.36 + (self.round * 0.072)
                    noise = random.uniform(-0.02, 0.04)
                    value = min(0.96, base_value + noise)
                elif metric == "auc":
                    base_value = 0.55 + (self.round * 0.06)
                    noise = random.uniform(-0.02, 0.03)
                    value = min(0.98, base_value + noise)
                elif metric == "r2_score":
                    base_value = 0.2 + (self.round * 0.09)
                    noise = random.uniform(-0.03, 0.05)
                    value = min(0.95, base_value + noise)

                metrics_report[metric] = round(value, 3)
            else:
                # Unknown metric - provide a default improving value
                base_value = 0.5 + (self.round * 0.05)
                noise = random.uniform(-0.02, 0.03)
                metrics_report[metric] = round(min(0.99, base_value + noise), 3)

        print(f"Round {self.round} metrics: {metrics_report}")
        return metrics_report

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
            X = np.load(
                os.path.join("data", f"X_{self.session_id}.npy"), allow_pickle=True
            )
            Y_test = np.load(
                os.path.join("data", f"Y_{self.session_id}.npy"), allow_pickle=True
            )

            (
                print("X : ", X.shape, X.dtype)
                if isinstance(X, np.ndarray)
                else print("X : ", len(X), type(X))
            )
            # Normalize dataset formats: allow object arrays of shape (n,1) with string rows
            try:
                first_elem = (
                    X[0]
                    if not (
                        isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 1
                    )
                    else X[0, 0]
                )
            except Exception:
                first_elem = None

            if isinstance(first_elem, str):
                print("Parsing string rows into individual pixel values")
                if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 1:
                    strings = [row[0] for row in X]
                else:
                    strings = list(X)
                X = np.array(
                    [
                        np.fromstring(s.strip(), sep=",", dtype=np.float32)
                        for s in strings
                    ],
                    dtype=np.float32,
                )
                # Optional reshape if input_shape is provided
                model_info = (
                    self.model_config.get("model_info", {})
                    if isinstance(self.model_config, dict)
                    else {}
                )
                input_shape = model_info.get("input_shape")
                if input_shape is not None:
                    if isinstance(input_shape, str):
                        # Handle string format like '(150,150,3)'
                        import ast

                        input_shape = ast.literal_eval(input_shape)
                    print(f"Parsed input_shape: {input_shape}")
                    try:
                        X = X.reshape(-1, *input_shape)
                    except Exception as reshape_err:
                        print(
                            f"Reshape failed with error {reshape_err}. Keeping flat features."
                        )

            print(f"X shape: {X.shape}")
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            return
        # calculate metrics from calculate_metrics function in metrics.py
        # round_results = calculate_metrics(Y_test, Y_pred, self.metrics)
        print("Metrics: ", self.metrics)

        # Temporary evaluate function that simulates improving metrics over rounds
        metrics_report = self.model.evaluate(X, Y_test, self.metrics)
        # metrics_report = self.temporary_evaluate_function()
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
