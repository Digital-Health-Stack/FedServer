import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings


class CustomSVR:
    """
    Linear Support Vector Regression (SVR) implementation tailored for federated learning.

    Design goals:
    - Linear kernel only (parameter sharing via weights/intercept is well-defined)
    - Stable sklearn-first training with StandardScaler
    - JSON-serializable parameter exchange (weights/biases as lists)

    API is aligned with other models in this codebase:
      - __init__(config=None, ...)
      - fit(X, y)
      - predict(X)
      - update_parameters(global_parameters)
      - get_parameters()
      - evaluate(X, y, metrics)
    """

    def __init__(
        self,
        config=None,
        C=1.0,
        epsilon=0.1,
        kernel="linear",
        random_state=42,
    ):
        def _to_float(value, default):
            try:
                return float(value)
            except Exception:
                return float(default)

        def _to_str(value, default):
            try:
                return str(value)
            except Exception:
                return str(default)

        if isinstance(config, dict):
            self.C = _to_float(config.get("C", C), C)
            self.epsilon = _to_float(config.get("epsilon", epsilon), epsilon)
            # Force linear kernel, but keep configurable for forward compatibility
            self.kernel = (
                "linear" if config.get("kernel", kernel) != "rbf" else "linear"
            )
            # random_state not used by sklearn.svm.SVR, but kept for API parity
            self.random_state = random_state
        else:
            self.C = _to_float(C, 1.0)
            self.epsilon = _to_float(epsilon, 0.1)
            self.kernel = _to_str(kernel, "linear")
            self.random_state = int(random_state)

        # Trained artifacts
        self.scaler = None
        self.sklearn_model = None
        self.use_sklearn = True

        # Parameters for federated sharing (linear kernel only)
        self.weights = None  # coef_
        self.biases = None  # intercept_

    # --------------------------
    # Training
    # --------------------------
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        # Scale features only (target scaling generally not used for SVR here)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        svr_params = {
            "kernel": "linear",
            "C": self.C,
            "epsilon": self.epsilon,
        }
        self.sklearn_model = SVR(**svr_params)
        self.sklearn_model.fit(X_scaled, y)
        self.use_sklearn = True

        # Extract linear parameters
        if hasattr(self.sklearn_model, "coef_"):
            self.weights = np.array(self.sklearn_model.coef_).ravel()
        else:
            # Some versions expose 1D coef_ as shape (n_features,)
            try:
                self.weights = np.array(self.sklearn_model.coef_)
            except Exception:
                self.weights = None
        if hasattr(self.sklearn_model, "intercept_"):
            self.biases = float(np.array(self.sklearn_model.intercept_).flatten()[0])
        else:
            self.biases = 0.0

    # --------------------------
    # Inference
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if (
            self.sklearn_model is not None
            and self.scaler is not None
            and self.use_sklearn
        ):
            X_scaled = self.scaler.transform(X)
            return self.sklearn_model.predict(X_scaled)

        if self.weights is None or self.biases is None:
            warnings.warn("Model has not been trained yet...NONE weights and biases.")
            return np.zeros(X.shape[0])

        w = np.array(self.weights).ravel()
        b = float(self.biases)
        return X.dot(w) + b

    # --------------------------
    # Federated parameter exchange
    # --------------------------
    def update_parameters(self, global_parameters):
        if global_parameters is None:
            return

        if "C" in global_parameters:
            self.C = float(global_parameters["C"])
        if "epsilon" in global_parameters:
            self.epsilon = float(global_parameters["epsilon"])

        if "weights" in global_parameters and "biases" in global_parameters:
            self.weights = np.array(global_parameters["weights"]).ravel()
            self.biases = float(np.array(global_parameters["biases"]).flatten()[0])
            # Switch to manual path when loading raw params
            self.use_sklearn = False

    def get_parameters(self):
        if self.use_sklearn and self.sklearn_model is not None:
            w = None
            b = None
            try:
                w = np.array(self.sklearn_model.coef_).ravel().tolist()
            except Exception:
                w = self.weights.tolist() if self.weights is not None else None
            try:
                b = float(np.array(self.sklearn_model.intercept_).flatten()[0])
            except Exception:
                b = float(self.biases) if self.biases is not None else None
            return {
                "weights": w,
                "biases": b,
            }

        if self.weights is None or self.biases is None:
            return {"weights": None, "biases": None}

        return {
            "weights": np.array(self.weights).ravel().tolist(),
            "biases": float(np.array(self.biases).flatten()[0]),
        }

    # --------------------------
    # Evaluation
    # --------------------------
    def evaluate(self, X, y, metrics):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        y_pred = self.predict(X)
        results = {}
        for metric in metrics or []:
            name = str(metric).lower()
            if name == "mse":
                results["mse"] = float(np.mean((y - y_pred) ** 2))
            elif name == "mae":
                results["mae"] = float(np.mean(np.abs(y - y_pred)))
            elif name == "r2" or name == "r2_score":
                # Compute coefficient of determination
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                results["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        return results
