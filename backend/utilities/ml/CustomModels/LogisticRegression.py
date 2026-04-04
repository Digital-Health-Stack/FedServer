import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class LogisticRegression:
    def __init__(self, config=None, lr=0.01, n_iters=1000):
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

        if isinstance(config, dict):
            self.lr = _to_float(config.get("lr", lr), lr)
            self.n_iters = _to_int(config.get("n_iters", n_iters), n_iters)
        else:
            self.lr = _to_float(lr, 0.01)
            self.n_iters = _to_int(n_iters, 1000)
        self.weights = None  # weights (manual model)
        self.bias = None  # bias (manual model)
        self.sklearn_model = None
        self.x_scaler = None
        self.use_sklearn = True  # flag: which one to use in predict

    # --------------------------
    # SIGMOID ACTIVATION FUNCTION
    # --------------------------
    def _sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    # --------------------------
    # OLD FIT (manual gradient descent)
    # --------------------------
    def fit(self, X, y):
        # Always prefer sklearn training for stability
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        try:
            self.fit_sklearn(X, y)
            return
        except Exception:
            # Manual gradient descent fallback
            pass

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Forward pass
            linear_pred = X.dot(self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            # Compute cost (cross-entropy loss)
            cost = (-1 / n_samples) * np.sum(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )

            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        self.use_sklearn = False  # mark active model

    # --------------------------
    # NEW FIT (scikit-learn pipeline)
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        # Fit scaler for X
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)

        # Fit classifier on scaled data
        clf = SGDClassifier(
            loss="log_loss",  # logistic regression loss
            learning_rate="constant",
            eta0=self.lr,
            max_iter=self.n_iters,
            tol=None,
            penalty=None,
            random_state=42,
        )
        clf.fit(X_scaled, y)
        self.sklearn_model = clf
        self.use_sklearn = True  # mark active model

    # --------------------------
    # PREDICT (choose based on flag)
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.sklearn_model is not None and self.x_scaler is not None:
            X_scaled = self.x_scaler.transform(X)
            # Return probabilities for class 1
            return self.sklearn_model.predict_proba(X_scaled)[:, 1]
        else:
            # Manual path
            if self.weights is None or self.bias is None:
                return np.zeros(X.shape[0])

            linear_pred = X.dot(self.weights) + self.bias
            return self._sigmoid(linear_pred)

    # --------------------------
    # PREDICT CLASSES
    # --------------------------
    def predict_classes(self, X, threshold=0.5):
        """Predict binary classes based on probability threshold"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)

    # --------------------------
    # UPDATE PARAMS
    # --------------------------
    def update_parameters(self, global_parameters):
        # Allow updating hyperparameters directly
        if global_parameters is not None:
            self.lr = float(global_parameters["learning_rate"])
            self.n_iters = int(global_parameters["iterations"])

        # Allow loading model parameters from server
        if (
            global_parameters is not None
            and "weights" in global_parameters
            and "bias" in global_parameters
        ):
            weights = global_parameters["weights"]
            bias = global_parameters["bias"]

            # weights may come as list/array (multi-feature) or simple float
            if weights is None:
                self.weights = None
            else:
                weights_arr = np.array(weights).ravel()
                self.weights = weights_arr
            self.bias = float(np.array(bias).flatten()[0]) if bias is not None else 0.0

            # prefer manual path when loading raw params
            self.use_sklearn = False

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        if self.sklearn_model is not None and self.x_scaler is not None:
            # Model in scaled space
            weights_scaled = np.array(self.sklearn_model.coef_).ravel()
            bias_scaled = float(np.array(self.sklearn_model.intercept_).flatten()[0])

            x_mean_vec = np.array(self.x_scaler.mean_).ravel()
            if hasattr(self.x_scaler, "scale_"):
                x_scale_vec = np.array(self.x_scaler.scale_).ravel()
            else:
                x_scale_vec = np.ones_like(x_mean_vec)

            safe_x_scale_vec = np.where(x_scale_vec == 0, 1.0, x_scale_vec)

            # Convert to original space:
            # For logistic regression: sigmoid(w_scaled · x_scaled + b_scaled)
            # where x_scaled = (x - μ_x) / σ_x
            # So: sigmoid(w_scaled · (x - μ_x) / σ_x + b_scaled)
            # = sigmoid((w_scaled / σ_x) · x + (b_scaled - w_scaled · μ_x / σ_x))
            weights_original = weights_scaled / safe_x_scale_vec
            bias_original = bias_scaled - np.sum(
                weights_scaled * x_mean_vec / safe_x_scale_vec
            )

            return {
                "weights": [float(v) for v in weights_original.tolist()],
                "bias": float(bias_original),
                "learning_rate": float(self.lr),
                "iterations": int(self.n_iters),
            }

        # Manual or uninitialized sklearn model -> return raw params
        if self.weights is None:
            weights_out = [0.0]
        elif isinstance(self.weights, np.ndarray):  # multi-feature case
            weights_out = [float(v) for v in self.weights.ravel().tolist()]
        else:  # scalar
            weights_out = [float(self.weights)]

        bias_out = float(self.bias) if self.bias is not None else 0.0

        return {
            "weights": weights_out,
            "bias": bias_out,
            "learning_rate": float(self.lr),
            "iterations": int(self.n_iters),
        }

    def evaluate(self, X, y, metrics):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        print("Metrics: ", metrics)

        y_prob = self.predict(X)
        y_pred = self.predict_classes(X)

        results = {}
        for metric in metrics or []:
            name = metric.lower()
            if name == "accuracy":
                results["accuracy"] = float(np.mean(y == y_pred))
            elif name == "precision":
                tp = np.sum((y == 1) & (y_pred == 1))
                fp = np.sum((y == 0) & (y_pred == 1))
                results["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            elif name == "recall":
                tp = np.sum((y == 1) & (y_pred == 1))
                fn = np.sum((y == 1) & (y_pred == 0))
                results["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            elif name == "f1":
                tp = np.sum((y == 1) & (y_pred == 1))
                fp = np.sum((y == 0) & (y_pred == 1))
                fn = np.sum((y == 1) & (y_pred == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                results["f1"] = (
                    float(2 * precision * recall / (precision + recall))
                    if (precision + recall) > 0
                    else 0.0
                )
            elif name == "log_loss":
                # Cross-entropy loss
                y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                results["log_loss"] = float(
                    -np.mean(
                        y * np.log(y_prob_clipped)
                        + (1 - y) * np.log(1 - y_prob_clipped)
                    )
                )
            else:
                pass
        return results
