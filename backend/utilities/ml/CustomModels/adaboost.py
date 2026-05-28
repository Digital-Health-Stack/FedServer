import numpy as np
from sklearn.preprocessing import StandardScaler


class AdaBoost:
    def __init__(self, config=None, n_estimators=50, learning_rate=1.0):
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
        else:
            self.n_estimators = _to_int(n_estimators, 50)
            self.learning_rate = _to_float(learning_rate, 1.0)

        self.sklearn_model = None
        self.x_scaler = None
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
            print(f"[DEBUG] sklearn fit failed: {e}, falling back to majority class")
        self.majority_class = int(np.bincount(y.astype(int)).argmax())
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        clf.fit(X_scaled, y)
        self.sklearn_model = clf
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = clf.feature_importances_.tolist()
        self.use_sklearn = True

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
        if hasattr(self, "majority_class"):
            return np.full(X.shape[0], float(self.majority_class))
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
            if "n_estimators" in global_parameters:
                self.n_estimators = int(global_parameters["n_estimators"])
            if "learning_rate" in global_parameters:
                self.learning_rate = float(global_parameters["learning_rate"])
            if "feature_importances" in global_parameters and global_parameters["feature_importances"] is not None:
                self.feature_importances_ = global_parameters["feature_importances"]

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        return {
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "feature_importances": self.feature_importances_,
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
