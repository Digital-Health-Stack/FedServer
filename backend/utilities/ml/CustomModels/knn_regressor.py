import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNRegressor:
    def __init__(self, config=None, n_neighbors=5, weights="uniform", metric="minkowski"):
        def _to_int(value, default):
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return int(default)

        if isinstance(config, dict):
            self.n_neighbors = _to_int(config.get("n_neighbors", n_neighbors), n_neighbors)
            self.weights = config.get("weights", weights)
            self.metric = config.get("metric", metric)
        else:
            self.n_neighbors = _to_int(n_neighbors, 5)
            self.weights = weights
            self.metric = metric

        self.X_train = None
        self.y_train = None
        self.sklearn_model = None
        self.x_scaler = None
        self.y_scaler = None
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
            print(f"[DEBUG] sklearn fit failed: {e}, falling back to manual KNN")
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_train = self.x_scaler.fit_transform(X)
        self.y_train = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        from sklearn.neighbors import KNeighborsRegressor
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()
        reg = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        reg.fit(X_scaled, y_scaled)
        self.sklearn_model = reg
        self.use_sklearn = True

    # --------------------------
    # MANUAL PREDICT HELPER
    # --------------------------
    def _manual_predict(self, X):
        X_scaled = self.x_scaler.transform(X)
        n_samples = X_scaled.shape[0]
        preds_scaled = np.zeros(n_samples)
        for i, x in enumerate(X_scaled):
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(dists)[: self.n_neighbors]
            preds_scaled[i] = np.mean(self.y_train[k_idx])
        return self.y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

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
        if self.X_train is not None:
            return self._manual_predict(X)
        return np.zeros(X.shape[0])

    # --------------------------
    # UPDATE PARAMS
    # --------------------------
    def update_parameters(self, global_parameters):
        if global_parameters is not None:
            if "n_neighbors" in global_parameters:
                self.n_neighbors = int(global_parameters["n_neighbors"])
            if "weights" in global_parameters:
                self.weights = global_parameters["weights"]
            if "metric" in global_parameters:
                self.metric = global_parameters["metric"]

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        return {
            "n_neighbors": int(self.n_neighbors),
            "weights": self.weights,
            "metric": self.metric,
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
