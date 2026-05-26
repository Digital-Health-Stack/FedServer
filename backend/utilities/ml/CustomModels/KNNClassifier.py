import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNClassifier:
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
        # Manual KNN: just store training data
        self.x_scaler = StandardScaler()
        self.X_train = self.x_scaler.fit_transform(X)
        self.y_train = y.copy()
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        from sklearn.neighbors import KNeighborsClassifier
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        clf = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        clf.fit(X_scaled, y)
        self.sklearn_model = clf
        self.use_sklearn = True

    # --------------------------
    # MANUAL PREDICT HELPER
    # --------------------------
    def _manual_predict_proba(self, X):
        X_scaled = self.x_scaler.transform(X)
        n_samples = X_scaled.shape[0]
        probs = np.zeros(n_samples)
        for i, x in enumerate(X_scaled):
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(dists)[: self.n_neighbors]
            k_labels = self.y_train[k_idx]
            probs[i] = np.mean(k_labels == 1)
        return probs

    # --------------------------
    # PREDICT
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.sklearn_model is not None and self.x_scaler is not None:
            X_scaled = self.x_scaler.transform(X)
            return self.sklearn_model.predict_proba(X_scaled)[:, 1]
        if self.X_train is not None:
            return self._manual_predict_proba(X)
        return np.zeros(X.shape[0])

    # --------------------------
    # PREDICT CLASSES
    # --------------------------
    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)

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
            elif name in ("f1_score", "f1"):
                tp = np.sum((y == 1) & (y_pred == 1))
                fp = np.sum((y == 0) & (y_pred == 1))
                fn = np.sum((y == 1) & (y_pred == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                results["f1_score"] = (
                    float(2 * precision * recall / (precision + recall))
                    if (precision + recall) > 0 else 0.0
                )
            elif name == "log_loss":
                y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                results["log_loss"] = float(
                    -np.mean(y * np.log(y_prob_clipped) + (1 - y) * np.log(1 - y_prob_clipped))
                )
        return results