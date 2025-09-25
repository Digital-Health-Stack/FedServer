import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


class MultiLayerPerceptron:
    """
    Scikit-learn based MLP supporting both classification and regression.

    API is aligned with CustomSVM and LinearRegression:
    - __init__(config=None, ...)
    - fit(X, y)
    - predict(X)
    - update_parameters(global_parameters)
    - get_parameters()
    - evaluate(X, y, metrics)

    Federated parameter exchange uses JSON-serializable lists
    for weights (coefs_) and biases (intercepts_).
    """

    def __init__(
        self,
        config=None,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        task_type="classification",  # or "regression"
        learning_rate=0.001,
        max_iter=200,
        alpha=0.0001,  # L2 regularization
        random_state=42,
        num_classes=None,
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

        def _to_tuple_of_ints(value, default):
            if value is None:
                return tuple(default)
            if isinstance(value, (list, tuple)):
                try:
                    return tuple(int(v) for v in value)
                except Exception:
                    return tuple(default)
            if isinstance(value, str):
                try:
                    parts = [
                        p.strip()
                        for p in value.replace("(", "").replace(")", "").split(",")
                        if p.strip() != ""
                    ]
                    return tuple(int(p) for p in parts)
                except Exception:
                    return tuple(default)
            return tuple(default)

        def _to_str(value, default):
            try:
                return str(value)
            except Exception:
                return str(default)

        if isinstance(config, dict):
            self.hidden_layer_sizes = _to_tuple_of_ints(
                config.get("hidden_layer_sizes", hidden_layer_sizes), hidden_layer_sizes
            )
            self.activation = _to_str(config.get("activation", activation), activation)
            self.task_type = _to_str(config.get("task_type", task_type), task_type)
            self.learning_rate = _to_float(
                config.get("learning_rate", learning_rate), learning_rate
            )
            self.max_iter = _to_int(config.get("max_iter", max_iter), max_iter)
            self.alpha = _to_float(config.get("alpha", alpha), alpha)
            self.random_state = _to_int(
                config.get("random_state", random_state), random_state
            )
            # Optional number of classes for classification
            self.num_classes = config.get("num_classes", num_classes)
            try:
                if isinstance(self.num_classes, str):
                    self.num_classes = int(self.num_classes)
            except Exception:
                self.num_classes = num_classes
        else:
            self.hidden_layer_sizes = tuple(hidden_layer_sizes)
            self.activation = str(activation)
            self.task_type = str(task_type)
            self.learning_rate = float(learning_rate)
            self.max_iter = int(max_iter)
            self.alpha = float(alpha)
            self.random_state = int(random_state)
            self.num_classes = num_classes

        # Model and scaling
        self.scaler = None
        self.sklearn_model = None
        self.use_sklearn = True
        # Cached scaler stats for transmission
        self._scaler_mean = None
        self._scaler_scale = None

        # Store weights/biases for manual forward pass when needed
        self.weights = None  # list of 2D arrays (coefs_)
        self.biases = None  # list of 1D arrays (intercepts_)

    # --------------------------
    # Helpers
    # --------------------------
    @staticmethod
    def _parse_to_2d_array(X):
        """Accepts arrays or lists of comma-separated strings and returns 2D np.array."""
        print(f"X: {X}")
        print(f"type(X): {type(X)}")
        print(f"len(X): {len(X)}")
        try:
            print(f"isinstance(X[0], str): {isinstance(X[0], str)}")
        except Exception:
            pass
        # Case 1: Python list of comma-separated strings
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
            try:
                parsed = []
                for row in X:
                    # split on commas and convert to float
                    vals = [float(v.strip()) for v in row.split(",") if v.strip() != ""]
                    parsed.append(vals)
                return np.array(parsed, dtype=float)
            except Exception:
                pass
        # Case 2: NumPy object array where each row is a single string, shape (n, 1)
        if isinstance(X, np.ndarray) and X.dtype == object:
            try:
                if X.ndim == 2 and X.shape[1] == 1 and isinstance(X[0, 0], str):
                    parsed = []
                    for row in X:
                        s = row[0]
                        vals = [
                            float(v.strip()) for v in s.split(",") if v.strip() != ""
                        ]
                        parsed.append(vals)
                    return np.array(parsed, dtype=float)
                # NumPy 1D object array of strings
                if X.ndim == 1 and len(X) > 0 and isinstance(X[0], str):
                    parsed = []
                    for s in X:
                        vals = [
                            float(v.strip()) for v in s.split(",") if v.strip() != ""
                        ]
                        parsed.append(vals)
                    return np.array(parsed, dtype=float)
            except Exception:
                pass
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr.astype(float)

    @staticmethod
    def _activation_forward(Z, name):
        if name == "relu":
            return np.maximum(0, Z)
        if name == "tanh":
            return np.tanh(Z)
        if name in ("logistic", "sigmoid"):
            return 1.0 / (1.0 + np.exp(-Z))
        # identity
        return Z

    @staticmethod
    def _softmax(Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # --------------------------
    # Fit
    # --------------------------
    def fit(self, X, y):
        X = self._parse_to_2d_array(X)
        y = np.array(y).ravel()

        # Infer task or correct mismatches based on target distribution
        num_unique = len(np.unique(y)) if y.size > 0 else 0
        num_samples = y.shape[0]
        # integer-like check (allow small numeric noise)
        y_is_integer_like = np.all(np.isfinite(y)) and np.all(
            np.abs(y - np.round(y)) < 1e-8
        )
        looks_continuous = (not y_is_integer_like) or (
            num_unique > max(50, int(0.1 * max(1, num_samples)))
        )

        if self.task_type not in ("classification", "regression"):
            self.task_type = "regression" if looks_continuous else "classification"
        elif self.task_type == "classification" and looks_continuous:
            print(
                "[MLP] Detected continuous targets; switching task_type to 'regression'."
            )
            self.task_type = "regression"
        elif self.task_type == "classification" and y_is_integer_like:
            # make sure labels are ints for sklearn
            y = y.astype(int)

        # Set up scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        # Cache scaler statistics for parameter exchange
        try:
            self._scaler_mean = (
                self.scaler.mean_.astype(float)
                if hasattr(self.scaler, "mean_")
                else None
            )
            self._scaler_scale = (
                self.scaler.scale_.astype(float)
                if hasattr(self.scaler, "scale_")
                else None
            )
        except Exception:
            self._scaler_mean, self._scaler_scale = None, None

        # Create sklearn model with warm_start to allow weight injection
        common_kwargs = dict(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=(
                self.activation
                if self.activation in ("relu", "tanh", "logistic", "identity")
                else "relu"
            ),
            solver="adam",
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=self.random_state,
            warm_start=True,
        )

        if self.task_type == "classification":
            self.sklearn_model = MLPClassifier(**common_kwargs)
        else:
            self.sklearn_model = MLPRegressor(**common_kwargs)

        # If we have received global weights, initialize model with them
        has_external_params = (
            self.weights is not None
            and self.biases is not None
            and len(self.weights) > 0
            and len(self.biases) > 0
        )

        if has_external_params:
            try:
                # Perform a tiny dummy fit to initialize internal structures
                if self.task_type == "classification":
                    # Include all unique classes in dummy fit to avoid warm_start class mismatch
                    unique_classes = np.unique(y)
                    if len(unique_classes) <= len(X_scaled):
                        # Use one sample per class if possible
                        dummy_indices = []
                        for cls in unique_classes:
                            cls_indices = np.where(y == cls)[0]
                            if len(cls_indices) > 0:
                                dummy_indices.append(cls_indices[0])
                        dummy_y = y[dummy_indices]
                        dummy_X = X_scaled[dummy_indices]
                    else:
                        # Fallback to first few samples
                        dummy_y = y[: min(len(unique_classes), len(X_scaled))]
                        dummy_X = X_scaled[: len(dummy_y)]
                else:
                    dummy_y = y[:2]
                    dummy_X = X_scaled[:2]
                self.sklearn_model.fit(dummy_X, dummy_y)

                # Inject provided weights/biases if shapes match
                provided_W = [np.array(w, dtype=float) for w in self.weights]
                provided_b = [np.array(b, dtype=float) for b in self.biases]
                if (
                    len(provided_W) == len(self.sklearn_model.coefs_)
                    and all(
                        w.shape == self.sklearn_model.coefs_[i].shape
                        for i, w in enumerate(provided_W)
                    )
                    and len(provided_b) == len(self.sklearn_model.intercepts_)
                    and all(
                        b.shape == self.sklearn_model.intercepts_[i].shape
                        for i, b in enumerate(provided_b)
                    )
                ):
                    self.sklearn_model.coefs_ = provided_W
                    self.sklearn_model.intercepts_ = provided_b
                else:
                    warnings.warn(
                        "Provided weights/biases shapes do not match model; skipping injection."
                    )
            except Exception:
                # If anything fails, fall back to normal fit
                pass

        # Train (continues from injected weights if any because warm_start=True)
        self.sklearn_model.fit(X_scaled, y)
        self.use_sklearn = True

        # Cache parameters for potential manual forward
        if hasattr(self.sklearn_model, "coefs_") and hasattr(
            self.sklearn_model, "intercepts_"
        ):
            self.weights = [np.array(w) for w in self.sklearn_model.coefs_]
            self.biases = [np.array(b) for b in self.sklearn_model.intercepts_]

    # --------------------------
    # Predict
    # --------------------------
    def predict(self, X):
        X = self._parse_to_2d_array(X)

        # Sklearn path
        if (
            self.sklearn_model is not None
            and self.scaler is not None
            and self.use_sklearn
        ):
            X_scaled = self.scaler.transform(X)
            try:
                return self.sklearn_model.predict(X_scaled)
            except Exception:
                pass

        # Manual forward pass
        if self.weights is None or self.biases is None:
            warnings.warn("Model has not been trained yet; returning zeros.")
            return np.zeros(X.shape[0])

        # Apply cached scaler if present
        if self._scaler_mean is not None and self._scaler_scale is not None:
            try:
                X = (X - self._scaler_mean) / self._scaler_scale
            except Exception:
                pass

        A = X
        act_name = (
            self.activation
            if self.activation in ("relu", "tanh", "logistic", "identity")
            else "relu"
        )
        for layer_idx in range(len(self.weights)):
            Z = A.dot(self.weights[layer_idx]) + self.biases[layer_idx]
            is_last = layer_idx == (len(self.weights) - 1)
            if is_last:
                if self.task_type == "classification":
                    # Approximate sklearn behavior: binary -> sigmoid; multi-class -> softmax
                    if self.num_classes is not None and self.num_classes > 2:
                        A = self._softmax(Z)
                        return np.argmax(A, axis=1)
                    # binary
                    A = 1.0 / (1.0 + np.exp(-Z))
                    return (A.ravel() >= 0.5).astype(int)
                else:
                    # regression uses identity on output
                    A = Z
                    return A.ravel()
            else:
                A = self._activation_forward(Z, act_name)

        return A

    # --------------------------
    # Update/Get parameters
    # --------------------------
    def update_parameters(self, global_parameters):
        """Update model hyperparameters or raw weights/biases."""
        if global_parameters is None:
            return

        # Hyperparameters
        if "hidden_layer_sizes" in global_parameters:
            hls = global_parameters["hidden_layer_sizes"]
            if isinstance(hls, str):
                try:
                    parts = [
                        p.strip()
                        for p in hls.replace("(", "").replace(")", "").split(",")
                        if p.strip() != ""
                    ]
                    self.hidden_layer_sizes = tuple(int(p) for p in parts)
                except Exception:
                    pass
            elif isinstance(hls, (list, tuple)):
                try:
                    self.hidden_layer_sizes = tuple(int(v) for v in hls)
                except Exception:
                    pass

        if "activation" in global_parameters:
            self.activation = str(global_parameters["activation"]) or self.activation

        if "task_type" in global_parameters:
            tt = str(global_parameters["task_type"]).lower()
            if tt in ("classification", "regression"):
                self.task_type = tt

        if "learning_rate" in global_parameters:
            try:
                self.learning_rate = float(global_parameters["learning_rate"])
            except Exception:
                pass

        if "max_iter" in global_parameters:
            try:
                self.max_iter = int(global_parameters["max_iter"])
            except Exception:
                pass

        if "alpha" in global_parameters:
            try:
                self.alpha = float(global_parameters["alpha"])
            except Exception:
                pass

        if "num_classes" in global_parameters:
            try:
                self.num_classes = int(global_parameters["num_classes"])
            except Exception:
                self.num_classes = None

        # Raw parameters (weights/biases)
        if "weights" in global_parameters and "biases" in global_parameters:
            weights = global_parameters["weights"]
            biases = global_parameters["biases"]
            try:
                self.weights = [np.array(w, dtype=float) for w in weights]
                self.biases = [np.array(b, dtype=float) for b in biases]
                # Switch to manual mode when raw params are set
                self.use_sklearn = False
                self.sklearn_model = None
                # Optional scaler stats
                scaler_cfg = global_parameters.get("scaler") or {}
                mean = scaler_cfg.get("mean")
                scale = scaler_cfg.get("scale")
                if mean is not None and scale is not None:
                    self._scaler_mean = np.array(mean, dtype=float)
                    self._scaler_scale = np.array(scale, dtype=float)
                self.scaler = None
            except Exception:
                warnings.warn(
                    "Failed to load provided weights/biases; keeping existing model."
                )

    def get_parameters(self):
        """Return JSON-serializable weights and biases."""
        if (
            self.use_sklearn
            and self.sklearn_model is not None
            and hasattr(self.sklearn_model, "coefs_")
        ):
            return {
                "weights": [w.tolist() for w in self.sklearn_model.coefs_],
                "biases": [b.tolist() for b in self.sklearn_model.intercepts_],
                "scaler": {
                    "mean": (
                        self._scaler_mean.tolist()
                        if self._scaler_mean is not None
                        else None
                    ),
                    "scale": (
                        self._scaler_scale.tolist()
                        if self._scaler_scale is not None
                        else None
                    ),
                },
            }

        if self.weights is not None and self.biases is not None:
            return {
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "scaler": {
                    "mean": (
                        self._scaler_mean.tolist()
                        if isinstance(self._scaler_mean, np.ndarray)
                        else self._scaler_mean
                    ),
                    "scale": (
                        self._scaler_scale.tolist()
                        if isinstance(self._scaler_scale, np.ndarray)
                        else self._scaler_scale
                    ),
                },
            }

        return {"weights": None, "biases": None}

    # --------------------------
    # Evaluate
    # --------------------------
    def evaluate(self, X, y, metrics):
        X = self._parse_to_2d_array(X)
        y = np.array(y).ravel()

        y_pred = self.predict(X)
        results = {}
        print("Metrics: ", metrics)
        print("Evaluating: ")
        print("Y pred: ", y_pred)
        print("Y: ", y)
        for metric in metrics or []:
            name = str(metric).lower()
            print("Name: ", name, "Task type: ", self.task_type)
            if name == "accuracy" and self.task_type == "classification":
                results["accuracy"] = float(np.mean(y == y_pred))
            elif name in ("f1", "f1_score") and self.task_type == "classification":
                try:
                    from sklearn.metrics import f1_score

                    results["f1_score"] = float(f1_score(y, y_pred, average="weighted"))
                except Exception:
                    pass
            elif name == "precision" and self.task_type == "classification":
                try:
                    from sklearn.metrics import precision_score

                    results["precision"] = float(
                        precision_score(y, y_pred, average="weighted")
                    )
                except Exception:
                    pass
            elif name == "recall" and self.task_type == "classification":
                try:
                    from sklearn.metrics import recall_score

                    results["recall"] = float(
                        recall_score(y, y_pred, average="weighted")
                    )
                except Exception:
                    pass
            elif (
                name in ("mse", "mean_squared_error") and self.task_type == "regression"
            ):
                try:
                    from sklearn.metrics import mean_squared_error

                    results["mse"] = float(mean_squared_error(y, y_pred))
                except Exception:
                    pass
            elif (
                name in ("mae", "mean_absolute_error")
                and self.task_type == "regression"
            ):
                try:
                    from sklearn.metrics import mean_absolute_error

                    results["mae"] = float(mean_absolute_error(y, y_pred))
                except Exception:
                    pass
            elif name in ("r2", "r2_score") and self.task_type == "regression":
                try:
                    from sklearn.metrics import r2_score

                    results["r2"] = float(r2_score(y, y_pred))
                except Exception:
                    pass
            elif name == "rmse" and self.task_type == "regression":
                try:
                    from sklearn.metrics import mean_squared_error
                    import math

                    results["rmse"] = float(math.sqrt(mean_squared_error(y, y_pred)))
                except Exception:
                    pass
            elif name == "loss" and self.task_type == "regression":
                # Approximate loss using MSE when not otherwise specified
                try:
                    from sklearn.metrics import mean_squared_error

                    results["loss"] = float(mean_squared_error(y, y_pred))
                except Exception:
                    pass
        try:
            print(f"Results count: {len(results)}")
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error in printing you can ignore this: {e}")
        return results
