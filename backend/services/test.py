import numpy as np
import os
import json
from utilities.ml.CustomModels.LandMarkSVM import LandMarkSVM
from utilities.ml.CustomModels.CustomSVM import CustomSVM
from utilities.ml.CustomModels.LinearRegression import LinearRegression
from utilities.ml.CustomModels.MultiLayerPerceptron import MultiLayerPerceptron
from utilities.ml.CustomModels.CustomCNN import CustomCNN
from utilities.ml.CustomModels.LogisticRegression import LogisticRegression
from utilities.ml.CustomModels.DecisionTree import DecisionTree
from utilities.ml.CustomModels.RandomForest import RandomForest
from utilities.ml.CustomModels.CustomSVR import CustomSVR
from utilities.ml.CustomModels.XGBoostRegressor import XGBoostRegressor
from utilities.ml.CustomModels.LassoRegression import LassoRegression
from utilities.ml.CustomModels.RidgeRegression import RidgeRegression
from utilities.ml.CustomModels.KNNClassifier import KNNClassifier
from utilities.ml.CustomModels.knn_regressor import KNNRegressor
from utilities.ml.CustomModels.adaboost import AdaBoost
from utilities.ml.CustomModels.lightgbm_regressor import LightGBMRegressor
from utilities.ml.CustomModels.lightgbm_classifier import LightGBMClassifier
from utilities.ml.CustomModels.naive_bayes import NaiveBayes
from utilities.ml.CustomModels.random_forest_regressor import RandomForestRegressor
from utilities.ml.CustomModels.xgboost_classifier import XGBoostClassifier
from utilities.core.db import engine
from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession, FederatedTestResults

model_classes = {
    "LinearRegression": LinearRegression,
    "SVM": CustomSVM,
    "SVR": CustomSVR,
    "LandMarkSVM": LandMarkSVM,
    "multiLayerPerceptron": MultiLayerPerceptron,
    "CNN": CustomCNN,
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTree,
    "RandomForest": RandomForest,
    "XGBoostRegressor": XGBoostRegressor,
    "LassoRegression": LassoRegression,
    "RidgeRegression": RidgeRegression,
    "KNNClassifier": KNNClassifier,
    "KNNRegressor": KNNRegressor,
    "AdaBoost": AdaBoost,
    "LightGBMRegressor": LightGBMRegressor,
    "LightGBMClassifier": LightGBMClassifier,
    "NaiveBayes": NaiveBayes,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBoostClassifier": XGBoostClassifier,
}


def model_instance_from_config(modelConfig):
    try:
        model_name = modelConfig["model_name"]
        config = modelConfig["model_info"]
        model_class = model_classes.get(model_name)

        if model_class is None:
            raise ValueError(f"Unknown model: {model_name}")

        model_instance = model_class(config)
        return model_instance

    except Exception as e:
        print(f"Error creating model instance: {e}")
        return None


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
