import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

"""
Linear Support Vector Machine (SVM) implementation for federated learning.

This implementation is restricted to LINEAR kernels only for simplicity and federated learning compatibility.
It supports both binary and multi-class classification.

Key features:
- Linear SVM only (no kernel trick)
- Supports both sklearn and manual training modes
- Designed for federated learning parameter sharing
- Feature scaling included for better performance

Usage Notes:
(i) If a new client wants to use this class to update the pretrained model (parameters on some server),
    it is recommended to first fetch the parameters from the server to ensure dimension compatibility.

(ii) The dimension of weights on the server must be same as the weights of the model to be updated/sent to the server.

(iii) Manual training fallback is available if sklearn training fails.
"""


class CustomSVM:
    def __init__(
        self,
        config=None,
        C=1.0,
        max_iter=1000,
        lr=0.01,
        is_binary=True,
    ):
        def _to_float(value, default):
            try:
                return float(value)
            except Exception:
                return float(default)

        def _to_int(value, default):
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return int(default)

        def _to_str(value, default):
            try:
                return str(value)
            except Exception:
                return str(default)

        if isinstance(config, dict):
            self.C = _to_float(config.get("C", C), C)
            self.max_iter = _to_int(config.get("max_iter", max_iter), max_iter)
            self.lr = _to_float(config.get("lr", lr), lr)
            self.is_binary = str(config.get("is_binary", is_binary)).lower() == "true"

            # Handle weights_shape for backward compatibility
            weights_shape = config.get("weights_shape", None)
            if weights_shape and isinstance(weights_shape, str):
                try:
                    import ast

                    self.weights_shape = ast.literal_eval(weights_shape)
                except:
                    self.weights_shape = None
            else:
                self.weights_shape = weights_shape
        else:
            self.C = _to_float(C, 1.0)
            self.max_iter = _to_int(max_iter, 1000)
            self.lr = _to_float(lr, 0.01)
            self.is_binary = bool(is_binary)
            self.weights_shape = None

        # Force linear kernel only
        self.kernel = "linear"

        # Model parameters
        self.weights = None
        self.biases = None
        self.sklearn_model = None
        self.scaler = None
        self.use_sklearn = True  # Prefer sklearn for better performance

        # Initialize weights if shape is provided
        if self.weights_shape is not None:
            self.weights = np.zeros(self.weights_shape)
            if isinstance(self.weights_shape, tuple) and len(self.weights_shape) == 2:
                self.biases = np.zeros(self.weights_shape[0])
            else:
                self.biases = np.zeros(self.weights_shape)

    def fit_binary(self, X, y):
        self.is_binary = True
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        lr = self.lr
        n_iters = self.max_iter
        binary_y = np.where(y == 1, 1, -1)
        for epoch in range(n_iters):
            for idx, x in enumerate(X):
                decision = np.dot(x, weights) + bias
                if binary_y[idx] * decision < 1:
                    weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                    bias += lr * binary_y[idx]
                else:
                    weights += lr * (-2 * self.C * weights)
        self.weights = weights
        self.biases = bias

    def fit(self, X, y):
        # Always prefer sklearn training for stability and kernel support
        X = np.array(X)
        y = np.array(y).ravel()

        try:
            self.fit_sklearn(X, y)
            return
        except Exception as e:
            print(f"Sklearn training failed: {e}, falling back to manual training")
            # Manual gradient descent fallback

        # Manual training fallback
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        if n_classes == 2 or self.is_binary:
            self.fit_binary(X, y)
            return

        if self.weights is None and self.biases is None:
            self.weights = np.zeros((n_classes, n_features))
            self.biases = np.zeros(n_classes)

        for i in classes:
            i = int(i)
            binary_y = np.where(y == i, 1, -1)
            weights = self.weights[i]
            bias = self.biases[i]
            lr = self.lr
            n_iters = self.max_iter

            for _ in range(n_iters):
                for idx, x in enumerate(X):
                    decision = np.dot(x, weights) + bias
                    if binary_y[idx] * decision < 1:
                        weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                        bias += lr * binary_y[idx]
                    else:
                        weights += lr * (-2 * self.C * weights)

            self.weights[i] = weights
            self.biases[i] = bias

        self.use_sklearn = False

    def fit_sklearn(self, X, y):
        """Fit using scikit-learn Linear SVM"""
        X = np.array(X)
        y = np.array(y).ravel()

        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Configure Linear SVM parameters
        svm_params = {
            "C": self.C,
            "kernel": "linear",
            "max_iter": self.max_iter,
            "random_state": 42,
        }

        # Create and fit the model
        self.sklearn_model = SVC(**svm_params)
        self.sklearn_model.fit(X_scaled, y)
        self.use_sklearn = True

        # Extract weights and biases (always available for linear kernel)
        self.weights = self.sklearn_model.coef_
        self.biases = self.sklearn_model.intercept_

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Use sklearn model if available
        if (
            self.sklearn_model is not None
            and self.scaler is not None
            and self.use_sklearn
        ):
            X_scaled = self.scaler.transform(X)
            return self.sklearn_model.predict(X_scaled)

        # Manual prediction fallback
        if self.weights is None or self.biases is None:
            warnings.warn("Model has not been trained yet...NONE weights and biases.")
            return np.zeros(X.shape[0])

        decision_values = np.dot(X, self.weights.T) + self.biases

        if self.is_binary or (hasattr(self, "weights") and self.weights.ndim == 1):
            return np.where(decision_values < 0, 0, 1)
        return np.argmax(decision_values, axis=1)

    def update_parameters(self, global_parameters):
        """Update model parameters from server"""
        if global_parameters is None:
            return

        # Update hyperparameters if provided
        if "C" in global_parameters:
            self.C = float(global_parameters["C"])
        if "max_iter" in global_parameters:
            self.max_iter = int(global_parameters["max_iter"])
        if "lr" in global_parameters:
            self.lr = float(global_parameters["lr"])

        # Update model weights and biases
        if "weights" in global_parameters and "biases" in global_parameters:
            self.weights = np.array(global_parameters["weights"])
            self.biases = np.array(global_parameters["biases"])
            self.use_sklearn = False  # Switch to manual mode when loading raw params

    def get_parameters(self):
        """Get model parameters for federated learning"""
        if self.use_sklearn and self.sklearn_model is not None:
            # For linear SVM, weights are always available
            return {
                "weights": self.weights.tolist() if self.weights is not None else None,
                "biases": self.biases.tolist() if self.biases is not None else None,
            }

        # Manual model parameters
        if self.weights is None or self.biases is None:
            return {
                "weights": None,
                "biases": None,
            }

        return {
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

    def evaluate(self, X, y, metrics):
        """Evaluate model performance using specified metrics"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        y_pred = self.predict(X)
        results = {}

        for metric in metrics or []:
            name = metric.lower()
            if name == "accuracy":
                results["accuracy"] = float(np.mean(y == y_pred))
            elif name == "f1_score" or name == "f1":
                from sklearn.metrics import f1_score

                results["f1_score"] = float(f1_score(y, y_pred, average="weighted"))
            elif name == "precision":
                from sklearn.metrics import precision_score

                results["precision"] = float(
                    precision_score(y, y_pred, average="weighted")
                )
            elif name == "recall":
                from sklearn.metrics import recall_score

                results["recall"] = float(recall_score(y, y_pred, average="weighted"))
            elif name == "mse":
                from sklearn.metrics import mean_squared_error

                results["mse"] = float(mean_squared_error(y, y_pred))
            elif name == "mae":
                from sklearn.metrics import mean_absolute_error

                results["mae"] = float(mean_absolute_error(y, y_pred))
            elif name == "confusion_matrix":
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y, y_pred)
                results["confusion_matrix"] = cm.tolist()
        try:
            print(len(results))
            print(results.slice(0, 10))
        except Exception as e:
            print(f"Error in printing you can ignore this: {e}")
        return results
