import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class LinearRegression:
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
        self.m = None  # slope (manual model)
        self.c = None  # intercept (manual model)
        self.sklearn_model = None
        self.x_scaler = None
        self.y_scaler = None
        self.use_sklearn = True  # flag: which one to use in predict

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

        n = len(X)
        m = 0
        c = 0

        for _ in range(self.n_iters):
            y_pred = m * X.flatten() + c
            dm = (-2 / n) * np.sum((y - y_pred) * X.flatten())
            dc = (-2 / n) * np.sum(y - y_pred)
            m -= self.lr * dm
            c -= self.lr * dc

        self.m = m
        self.c = c
        self.use_sklearn = False  # mark active model

    # --------------------------
    # NEW FIT (scikit-learn pipeline)
    # --------------------------
    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        # Fit separate scalers for X and y
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        # Fit regressor on scaled data
        reg = SGDRegressor(
            learning_rate="constant",
            eta0=self.lr,
            max_iter=self.n_iters,
            tol=None,
            penalty=None,
            random_state=42,
        )
        reg.fit(X_scaled, y_scaled)
        self.sklearn_model = reg
        self.use_sklearn = True  # mark active model

    # --------------------------
    # PREDICT (choose based on flag)
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if (
            self.sklearn_model is not None
            and self.x_scaler is not None
            and self.y_scaler is not None
        ):
            X_scaled = self.x_scaler.transform(X)
            y_scaled_pred = self.sklearn_model.predict(X_scaled).reshape(-1, 1)
            y_pred = self.y_scaler.inverse_transform(y_scaled_pred).ravel()
            return y_pred
        else:
            # Manual path supports single or multi-feature if m is a vector
            if self.m is None or self.c is None:
                return np.zeros(X.shape[0])
            m_vec = np.array(self.m).ravel()
            if X.shape[1] == m_vec.shape[0]:
                return X.dot(m_vec) + float(self.c)
            if m_vec.size == 1 and X.shape[1] == 1:
                return m_vec[0] * X.flatten() + float(self.c)
            return np.zeros(X.shape[0])

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
            and "m" in global_parameters
            and "c" in global_parameters
        ):
            m = global_parameters["m"]
            c = global_parameters["c"]

            # coef may come as list/array (multi-feature) or simple float
            if m is None:
                self.m = None
            else:
                m_arr = np.array(m).ravel()
                self.m = m_arr if m_arr.size > 1 else float(m_arr[0])
            self.c = float(np.array(c).flatten()[0]) if c is not None else 0.0

            # prefer manual path when loading raw params
            self.use_sklearn = False

    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        if (
            self.sklearn_model is not None
            and self.x_scaler is not None
            and self.y_scaler is not None
        ):
            # Model in scaled space: y_s = a · x_s + b
            a_vec = np.array(self.sklearn_model.coef_).ravel()
            b = float(np.array(self.sklearn_model.intercept_).flatten()[0])

            x_mean_vec = np.array(self.x_scaler.mean_).ravel()
            if hasattr(self.x_scaler, "scale_"):
                x_scale_vec = np.array(self.x_scaler.scale_).ravel()
            else:
                x_scale_vec = np.ones_like(x_mean_vec)

            y_mean = float(np.array(self.y_scaler.mean_).flatten()[0])
            if hasattr(self.y_scaler, "scale_"):
                y_scale = float(np.array(self.y_scaler.scale_).flatten()[0])
            else:
                y_scale = 1.0

            safe_x_scale_vec = np.where(x_scale_vec == 0, 1.0, x_scale_vec)
            safe_y_scale = y_scale if y_scale != 0 else 1.0

            # Convert to original space:
            # y = (a · ((x - μ_x)/σ_x) + b) * σ_y + μ_y
            # m = (σ_y * a) / σ_x (element-wise)
            # c = σ_y * (b - Σ a_j * μ_xj / σ_xj) + μ_y
            m_vec = (safe_y_scale * a_vec) / safe_x_scale_vec
            c_raw = (
                safe_y_scale * (b - np.sum(a_vec * x_mean_vec / safe_x_scale_vec))
                + y_mean
            )

            return {
                "m": [float(v) for v in m_vec.tolist()],
                "c": float(c_raw),
                "learning_rate": float(self.lr),
                "iterations": int(self.n_iters),
            }

        # Manual or uninitialized sklearn model -> return scalar params
        # m = float(self.m) if self.m is not None else 0.0
        # c = float(self.c) if self.c is not None else 0.0
        # return {
        #     "m": float(m),
        #     "c": float(c),
        #     "learning_rate": float(self.lr),
        #     "iterations": int(self.n_iters),
        # }
        if self.m is None:
            m_out = 0.0
        elif isinstance(self.m, np.ndarray):  # multi-feature case
            m_out = [float(v) for v in self.m.ravel().tolist()]
        else:  # scalar
            m_out = float(self.m)

        c_out = float(self.c) if self.c is not None else 0.0

        return {
            "m": m_out,
            "c": c_out,
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
            else:
                pass
        return results
