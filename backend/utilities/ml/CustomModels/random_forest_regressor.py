import numpy as np
from sklearn.preprocessing import StandardScaler


class RandomForestRegressor:
    def __init__(self, config=None, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt"):
        def _to_int_or_none(value, default):
            if value is None or value == "None":
                return None
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return default

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
            self.max_depth = _to_int_or_none(config.get("max_depth", max_depth), max_depth)
            self.min_samples_split = _to_int(config.get("min_samples_split", min_samples_split), min_samples_split)
            self.max_features = config.get("max_features", max_features)
        else:
            self.n_estimators = _to_int(n_estimators, 100)
            self.max_depth = _to_int_or_none(max_depth, None)
            self.min_samples_split = _to_int(min_samples_split, 2)
            self.max_features = max_features

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
        from sklearn.ensemble import RandomForestRegressor as SKLearnRFR
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()
        reg = SKLearnRFR(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=42,
        )
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
            if "max_depth" in global_parameters:
                v = global_parameters["max_depth"]
                self.max_depth = None if v is None or v == "None" else int(v)
            if "min_samples_split" in global_parameters:
                self.min_samples_split = int(global_parameters["min_samples_split"])
            if "max_features" in global_parameters:
                self.max_features = global_parameters["max_features"]
            if "feature_importances" in global_parameters and global_parameters["feature_importances"] is not None:
                self.feature_importances_ = global_parameters["feature_importances"]

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_split": int(self.min_samples_split),
            "max_features": self.max_features,
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
