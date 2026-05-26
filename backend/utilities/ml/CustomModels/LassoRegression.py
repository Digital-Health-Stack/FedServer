import numpy as np
from sklearn.preprocessing import StandardScaler


class LassoRegression:
    def __init__(self, config=None, alpha=1.0, n_iters=1000, lr=0.01):
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
            self.alpha = _to_float(config.get("alpha", alpha), alpha)
            self.n_iters = _to_int(config.get("n_iters", n_iters), n_iters)
            self.lr = _to_float(config.get("lr", lr), lr)
        else:
            self.alpha = _to_float(alpha, 1.0)
            self.n_iters = _to_int(n_iters, 1000)
            self.lr = _to_float(lr, 0.01)

        self.weights = None
        self.bias = None
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
            print(f"[DEBUG] sklearn fit failed: {e}, falling back to manual coordinate descent")
        # Manual gradient descent with L1 (subgradient)
        n, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iters):
            y_pred = X.dot(self.weights) + self.bias
            dw = (-2 / n) * X.T.dot(y - y_pred) + self.alpha * np.sign(self.weights)
            db = (-2 / n) * np.sum(y - y_pred)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        self.use_sklearn = False

    # --------------------------
    # FIT SKLEARN
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        from sklearn.linear_model import Lasso
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()
        reg = Lasso(alpha=self.alpha, max_iter=self.n_iters)
        reg.fit(X_scaled, y_scaled)
        self.sklearn_model = reg
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
        if self.weights is None or self.bias is None:
            return np.zeros(X.shape[0])
        return X.dot(np.array(self.weights).ravel()) + float(self.bias)

    # --------------------------
    # UPDATE PARAMS
    # --------------------------
    def update_parameters(self, global_parameters):
        if global_parameters is not None:
            if "alpha" in global_parameters:
                self.alpha = float(global_parameters["alpha"])
            if "learning_rate" in global_parameters:
                self.lr = float(global_parameters["learning_rate"])
            if "iterations" in global_parameters:
                self.n_iters = int(global_parameters["iterations"])
            if "weights" in global_parameters and "bias" in global_parameters:
                w = global_parameters["weights"]
                b = global_parameters["bias"]
                self.weights = np.array(w).ravel() if w is not None else None
                self.bias = float(np.array(b).flatten()[0]) if b is not None else 0.0
                self.use_sklearn = False

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        if self.sklearn_model is not None and self.x_scaler is not None and self.y_scaler is not None:
            a_vec = np.array(self.sklearn_model.coef_).ravel()
            b = float(np.array(self.sklearn_model.intercept_).flatten()[0])
            x_mean = np.array(self.x_scaler.mean_).ravel()
            x_scale = np.where(np.array(self.x_scaler.scale_).ravel() == 0, 1.0, np.array(self.x_scaler.scale_).ravel())
            y_mean = float(self.y_scaler.mean_.flatten()[0])
            y_scale = float(self.y_scaler.scale_.flatten()[0]) if self.y_scaler.scale_.flatten()[0] != 0 else 1.0
            weights = (y_scale * a_vec) / x_scale
            bias = y_scale * (b - np.sum(a_vec * x_mean / x_scale)) + y_mean
            return {
                "weights": [float(v) for v in weights.tolist()],
                "bias": float(bias),
                "alpha": float(self.alpha),
                "learning_rate": float(self.lr),
                "iterations": int(self.n_iters),
            }
        w_out = [float(v) for v in np.array(self.weights).ravel().tolist()] if self.weights is not None else [0.0]
        b_out = float(self.bias) if self.bias is not None else 0.0
        return {
            "weights": w_out,
            "bias": b_out,
            "alpha": float(self.alpha),
            "learning_rate": float(self.lr),
            "iterations": int(self.n_iters),
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