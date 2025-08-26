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
        self.m = None   # slope (manual model)
        self.c = None   # intercept (manual model)
        self.sklearn_model = None
        self.use_sklearn = True  # flag: which one to use in predict
    
    # --------------------------
    # OLD FIT (manual gradient descent)
    # --------------------------
    def fit(self, X, y):
        # Always prefer sklearn training for stability
        X = np.array(X).reshape(-1, 1)
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
            dm = (-2/n) * np.sum((y - y_pred) * X.flatten())
            dc = (-2/n) * np.sum(y - y_pred)
            m -= self.lr * dm
            c -= self.lr * dc

        self.m = m
        self.c = c
        self.use_sklearn = False  # mark active model

    # --------------------------
    # NEW FIT (scikit-learn pipeline)
    # --------------------------
    def fit_sklearn(self, X, y):
        self.sklearn_model = make_pipeline(
            StandardScaler(),
            SGDRegressor(
                learning_rate="constant",
                eta0=self.lr,
                max_iter=self.n_iters,
                tol=None,
                penalty=None,
                random_state=42,
            )
        )
        self.sklearn_model.fit(X, y)
        self.use_sklearn = True  # mark active model
    
    # --------------------------
    # PREDICT (choose based on flag)
    # --------------------------
    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        if self.sklearn_model is not None:
            return self.sklearn_model.predict(X)
        else:
            if self.m is None or self.c is None:
                # Uninitialized manual parameters -> return zeros
                return np.zeros(X.shape[0])
            return self.m * X.flatten() + self.c
    
    # --------------------------
    # UPDATE PARAMS
    # --------------------------
    def update_parameters(self, parameters=None, lr=None, n_iters=None):
        # Allow updating hyperparameters directly
        if lr is not None:
            self.lr = float(lr)
        if n_iters is not None:
            self.n_iters = int(n_iters)

        # Allow loading model parameters from server
        if parameters is not None and "m" in parameters and "c" in parameters:
            m = parameters["m"]
            c = parameters["c"]

            # coef may come as list/array or simple float
            self.m = float(np.array(m).flatten()[0]) if m is not None else 0.0
            self.c = float(np.array(c).flatten()[0]) if c is not None else 0.0

            # prefer manual path when loading raw params
            self.use_sklearn = False
    
    # --------------------------
    # GET PARAMS
    # --------------------------
    def get_parameters(self):
        if self.sklearn_model is not None:
            reg = self.sklearn_model.named_steps["sgdregressor"]
            scaler = self.sklearn_model.named_steps["standardscaler"]
            a = float(np.array(reg.coef_).flatten()[0])
            b = float(np.array(reg.intercept_).flatten()[0])
            mean = float(np.array(scaler.mean_).flatten()[0])
            scale = float(np.array(scaler.scale_).flatten()[0]) if hasattr(scaler, "scale_") else 1.0
            # Convert from standardized space: y = a*((x-mean)/scale)+b => y = (a/scale)*x + (b - a*mean/scale)
            m_raw = a / (scale if scale != 0 else 1.0)
            c_raw = b - a * mean / (scale if scale != 0 else 1.0)
            return {
                "m": float(m_raw),
                "c": float(c_raw),
                "learning_rate": float(self.lr),
                "iterations": int(self.n_iters),
            }

        # Manual or uninitialized sklearn model -> return scalar params
        m = float(self.m) if self.m is not None else 0.0
        c = float(self.c) if self.c is not None else 0.0
        return {
            "m": float(m),
            "c": float(c),
            "learning_rate": float(self.lr),
            "iterations": int(self.n_iters),
        }

    def evaluate(self, X, y, metrics):
        X = np.array(X).reshape(-1, 1)
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
