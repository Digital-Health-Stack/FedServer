import numpy as np
from sklearn.preprocessing import StandardScaler


class NaiveBayes:
    def __init__(self, config=None, var_smoothing=1e-9):
        def _to_float(value, default):
            try:
                return float(value)
            except Exception:
                return float(default)

        if isinstance(config, dict):
            self.var_smoothing = _to_float(config.get("var_smoothing", var_smoothing), var_smoothing)
        else:
            self.var_smoothing = _to_float(var_smoothing, 1e-9)

        self.sklearn_model = None
        self.x_scaler = None
        # Manual model params
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None   # class means
        self.sigma_ = None   # class variances
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
            print(f"[DEBUG] sklearn fit failed: {e}, falling back to manual Gaussian NB")
        # Manual Gaussian Naive Bayes
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.class_prior_ = np.array([np.mean(y == c) for c in self.classes_])
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.sigma_ = np.array([X[y == c].var(axis=0) + self.var_smoothing for c in self.classes_])
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        from sklearn.naive_bayes import GaussianNB
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        clf = GaussianNB(var_smoothing=self.var_smoothing)
        clf.fit(X_scaled, y)
        self.sklearn_model = clf
        # Mirror internal params for get_parameters
        self.classes_ = clf.classes_.tolist()
        self.class_prior_ = clf.class_prior_.tolist()
        self.theta_ = clf.theta_.tolist()
        self.sigma_ = clf.var_.tolist()
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
        # Manual Gaussian NB predict
        if self.theta_ is None:
            return np.zeros(X.shape[0])
        theta = np.array(self.theta_)
        sigma = np.array(self.sigma_)
        log_priors = np.log(np.array(self.class_prior_) + 1e-15)
        log_likelihoods = []
        for k in range(len(self.classes_)):
            diff = X - theta[k]
            log_like = -0.5 * np.sum(np.log(2 * np.pi * sigma[k]) + (diff ** 2) / sigma[k], axis=1)
            log_likelihoods.append(log_like + log_priors[k])
        log_likelihoods = np.array(log_likelihoods).T  # (n_samples, n_classes)
        # Softmax for numerical stability
        log_likelihoods -= log_likelihoods.max(axis=1, keepdims=True)
        probs = np.exp(log_likelihoods)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

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
            if "var_smoothing" in global_parameters:
                self.var_smoothing = float(global_parameters["var_smoothing"])
            if "theta" in global_parameters and global_parameters["theta"] is not None:
                self.theta_ = global_parameters["theta"]
            if "sigma" in global_parameters and global_parameters["sigma"] is not None:
                self.sigma_ = global_parameters["sigma"]
            if "class_prior" in global_parameters and global_parameters["class_prior"] is not None:
                self.class_prior_ = global_parameters["class_prior"]
            if "classes" in global_parameters and global_parameters["classes"] is not None:
                self.classes_ = global_parameters["classes"]
            # If model params are loaded, use manual path
            if "theta" in global_parameters:
                self.use_sklearn = False

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        def _safe(v):
            try:
                arr = np.array(v)
                return np.where(np.isfinite(arr), arr, 0.0).tolist()
            except Exception:
                return v
        return {
            "var_smoothing": float(self.var_smoothing),
            "classes": _safe(self.classes_) if self.classes_ is not None else None,
            "class_prior": _safe(self.class_prior_) if self.class_prior_ is not None else None,
            "theta": _safe(self.theta_) if self.theta_ is not None else None,
            "sigma": _safe(self.sigma_) if self.sigma_ is not None else None,
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
