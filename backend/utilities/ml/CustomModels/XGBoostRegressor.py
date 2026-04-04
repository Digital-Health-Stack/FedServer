import numpy as np


class XGBoostRegressor:
    """
    XGBoost-based regressor with an API aligned to other custom models.

    Exposes:
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
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
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

        # Defaults
        if isinstance(config, dict):
            self.learning_rate = _to_float(
                config.get("learning_rate", learning_rate), learning_rate
            )
            self.n_estimators = _to_int(
                config.get("n_estimators", n_estimators), n_estimators
            )
            self.max_depth = _to_int(config.get("max_depth", max_depth), max_depth)
            self.subsample = _to_float(config.get("subsample", subsample), subsample)
            self.colsample_bytree = _to_float(
                config.get("colsample_bytree", colsample_bytree), colsample_bytree
            )
            self.random_state = _to_int(
                config.get("random_state", random_state), random_state
            )
        else:
            self.learning_rate = _to_float(learning_rate, 0.1)
            self.n_estimators = _to_int(n_estimators, 200)
            self.max_depth = _to_int(max_depth, 6)
            self.subsample = _to_float(subsample, 1.0)
            self.colsample_bytree = _to_float(colsample_bytree, 1.0)
            self.random_state = _to_int(random_state, 42)

        self.model = None

    # --------------------------
    # Fit
    # --------------------------
    def fit(self, X, y):
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise RuntimeError(
                "xgboost is required for XGBoostRegressor but is not installed"
            ) from e

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="reg:squarederror",
            tree_method="auto",
        )
        self.model.fit(X, y)

    # --------------------------
    # Predict
    # --------------------------
    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)

    # --------------------------
    # Parameters exchange (best-effort for federated interface)
    # --------------------------
    def update_parameters(self, global_parameters):
        if not global_parameters:
            return
        # Update hyperparameters if provided
        self.learning_rate = float(
            global_parameters.get("learning_rate", self.learning_rate)
        )
        self.n_estimators = int(
            global_parameters.get("n_estimators", self.n_estimators)
        )
        self.max_depth = int(global_parameters.get("max_depth", self.max_depth))
        self.subsample = float(global_parameters.get("subsample", self.subsample))
        self.colsample_bytree = float(
            global_parameters.get("colsample_bytree", self.colsample_bytree)
        )
        # Model internal trees are not easily serializable/mergeable; skip tree import

    def get_parameters(self):
        return {
            "learning_rate": float(self.learning_rate),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
        }

    # --------------------------
    # Evaluate
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
        return results
