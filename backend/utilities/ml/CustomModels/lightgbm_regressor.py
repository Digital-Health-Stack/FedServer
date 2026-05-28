import numpy as np
from sklearn.preprocessing import StandardScaler


class LightGBMRegressor:
    def __init__(self, config=None, n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31):
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
            self.n_estimators = _to_int(config.get("n_estimators", n_estimators), n_estimators)
            self.learning_rate = _to_float(config.get("learning_rate", learning_rate), learning_rate)
            self.max_depth = _to_int(config.get("max_depth", max_depth), max_depth)
            self.num_leaves = _to_int(config.get("num_leaves", num_leaves), num_leaves)
        else:
            self.n_estimators = _to_int(n_estimators, 100)
            self.learning_rate = _to_float(learning_rate, 0.1)
            self.max_depth = _to_int(max_depth, -1)
            self.num_leaves = _to_int(num_leaves, 31)

        self.sklearn_model = None
        self.x_scaler = None
        self.y_scaler = None
        self.feature_importances_ = None
        self.use_sklearn = True

    # --------------------------
    # FIT
    # --------------------------
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        try:
            self.fit_sklearn(X, y)
            return
        except Exception as e:
            print(f"[DEBUG] sklearn fit failed: {e}, falling back to mean predictor")
        self.mean_target = float(np.mean(y))
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        try:
            from lightgbm import LGBMRegressor
            reg = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            reg = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=max(self.max_depth, 3),
                random_state=42,
            )
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()
        reg.fit(X_scaled, y_scaled)
        self.sklearn_model = reg
        if hasattr(reg, "feature_importances_"):
            self.feature_importances_ = reg.feature_importances_.tolist()
        self.use_sklearn = True

    # --------------------------
    # PREDICT
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.sklearn_model is not None and self.x_scaler is not None and self.y_scaler is not None:
            X_scaled = self.x_scaler.transform(X)
            y_scaled_pred = self.sklearn_model.predict(X_scaled).reshape(-1, 1)
            return self.y_scaler.inverse_transform(y_scaled_pred).ravel()
        if hasattr(self, "mean_target"):
            return np.full(X.shape[0], self.mean_target)
        return np.zeros(X.shape[0])

    # --------------------------
    # UPDATE PARAMS
    # --------------------------
    def update_parameters(self, global_parameters):
        if global_parameters is not None:
            if "n_estimators" in global_parameters:
                self.n_estimators = int(global_parameters["n_estimators"])
            if "learning_rate" in global_parameters:
                self.learning_rate = float(global_parameters["learning_rate"])
            if "max_depth" in global_parameters:
                self.max_depth = int(global_parameters["max_depth"])
            if "num_leaves" in global_parameters:
                self.num_leaves = int(global_parameters["num_leaves"])
            if "feature_importances" in global_parameters and global_parameters["feature_importances"] is not None:
                self.feature_importances_ = global_parameters["feature_importances"]

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        return {
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "num_leaves": int(self.num_leaves),
            "feature_importances": self.feature_importances_,
        }

    def evaluate(self, X, y, metrics):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        print("Metrics: ", metrics)
        y_pred = self.predict(X)
        results = {}
        for metric in metrics or []:
            name = metric.lower()
            if name == "mse":
                results["mse"] = float(np.mean((y - y_pred) ** 2))
            elif name == "mae":
                results["mae"] = float(np.mean(np.abs(y - y_pred)))
        return results
