import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings


class CustomSVR:
    """
    Linear Support Vector Regression (SVR) implementation tailored for federated learning.

    Design goals:
    - Linear kernel only (parameter sharing via weights/intercept is well-defined)
    - Stable sklearn-first training with StandardScaler
    - JSON-serializable parameter exchange (weights/biases as lists)

    API is aligned with other models in this codebase:
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
        C=1.0,
        epsilon=0.1,
        kernel="linear",
        gamma="scale",
        degree=3,
        coef0=0.0,
        random_state=42,
    ):
        def _to_float(value, default):
            try:
                return float(value)
            except Exception:
                return float(default)

        def _to_str(value, default):
            try:
                return str(value)
            except Exception:
                return str(default)

        if isinstance(config, dict):
            self.C = _to_float(config.get("C", C), C)
            self.epsilon = _to_float(config.get("epsilon", epsilon), epsilon)
            # Allow linear/rbf/poly; default to linear if unknown
            kernel_in = _to_str(config.get("kernel", kernel), kernel).lower()
            self.kernel = (
                kernel_in if kernel_in in {"linear", "rbf", "poly"} else "linear"
            )
            # gamma supports numeric, 'scale', or 'auto'
            gamma_in = config.get("gamma", gamma)
            try:
                self.gamma = float(gamma_in)
            except Exception:
                self.gamma = _to_str(gamma_in, "scale")
            self.degree = int(config.get("degree", degree))
            self.coef0 = _to_float(config.get("coef0", coef0), coef0)
            # random_state not used by sklearn.svm.SVR, but kept for API parity
            self.random_state = random_state
        else:
            self.C = _to_float(C, 1.0)
            self.epsilon = _to_float(epsilon, 0.1)
            self.kernel = _to_str(kernel, "linear")
            self.gamma = gamma
            self.degree = int(degree)
            self.coef0 = float(coef0)
            self.random_state = int(random_state)

        # Trained artifacts
        self.scaler = None
        self.sklearn_model = None
        self.use_sklearn = True

        # Parameters for federated sharing
        # Linear
        self.weights = None  # coef_
        self.biases = None  # intercept_
        # Non-linear kernels
        self.support_vectors = None  # SVs in scaled space
        self.dual_coef = None  # alpha vector
        self.gamma_resolved = None  # numeric gamma actually used

    # --------------------------
    # Training
    # --------------------------
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        # Scale features only (target scaling generally not used for SVR here)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        svr_params = {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
        }
        self.sklearn_model = SVR(**svr_params)
        self.sklearn_model.fit(X_scaled, y)
        self.use_sklearn = True

        # Extract parameters for sharing
        if self.kernel == "linear":
            if hasattr(self.sklearn_model, "coef_"):
                self.weights = np.array(self.sklearn_model.coef_).ravel()
            else:
                try:
                    self.weights = np.array(self.sklearn_model.coef_)
                except Exception:
                    self.weights = None
            if hasattr(self.sklearn_model, "intercept_"):
                self.biases = float(
                    np.array(self.sklearn_model.intercept_).flatten()[0]
                )
            else:
                self.biases = 0.0
        else:
            # Store kernel parameters to enable manual prediction across clients
            try:
                self.support_vectors = np.array(self.sklearn_model.support_vectors_)
                self.dual_coef = np.array(self.sklearn_model.dual_coef_).ravel()
                self.biases = float(
                    np.array(self.sklearn_model.intercept_).flatten()[0]
                )
                self.gamma_resolved = getattr(self.sklearn_model, "_gamma", None)
            except Exception:
                self.support_vectors = None
                self.dual_coef = None
                self.biases = 0.0
                self.gamma_resolved = None
            self.weights = None

    # --------------------------
    # Inference
    # --------------------------
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if (
            self.sklearn_model is not None
            and self.scaler is not None
            and self.use_sklearn
        ):
            X_scaled = self.scaler.transform(X)
            return self.sklearn_model.predict(X_scaled)

        # Manual prediction path
        if self.kernel == "linear":
            if self.weights is None or self.biases is None:
                warnings.warn(
                    "Model has not been trained yet...NONE weights and biases."
                )
                return np.zeros(X.shape[0])
            w = np.array(self.weights).ravel()
            b = float(self.biases)
            return X.dot(w) + b

        # Non-linear kernels require stored SV parameters and scaler
        if (
            self.scaler is None
            or self.support_vectors is None
            or self.dual_coef is None
            or self.biases is None
        ):
            warnings.warn("Model parameters incomplete for manual kernel prediction.")
            return np.zeros(X.shape[0])

        X_scaled = self.scaler.transform(X)
        SV = np.array(self.support_vectors)
        alpha = np.array(self.dual_coef).ravel()
        b = float(self.biases)

        if self.kernel == "rbf":
            gamma_val = self.gamma_resolved
            if gamma_val is None:
                try:
                    gamma_val = float(self.gamma)
                except Exception:
                    gamma_val = 0.1
            x_norm = np.sum(X_scaled**2, axis=1)[:, None]
            sv_norm = np.sum(SV**2, axis=1)[None, :]
            cross = X_scaled @ SV.T
            K = np.exp(-gamma_val * (x_norm + sv_norm - 2 * cross))
        elif self.kernel == "poly":
            try:
                gamma_val = float(self.gamma)
            except Exception:
                gamma_val = 1.0
            K = (gamma_val * (X_scaled @ SV.T) + float(self.coef0)) ** int(self.degree)
        else:
            return np.zeros(X.shape[0])

        return K @ alpha + b

    # --------------------------
    # Federated parameter exchange
    # --------------------------
    def update_parameters(self, global_parameters):
        if global_parameters is None:
            return

        if "C" in global_parameters:
            self.C = float(global_parameters["C"])
        if "epsilon" in global_parameters:
            self.epsilon = float(global_parameters["epsilon"])
        if "kernel" in global_parameters:
            k = str(global_parameters["kernel"]).lower()
            self.kernel = k if k in {"linear", "rbf", "poly"} else self.kernel
        if "gamma" in global_parameters:
            try:
                self.gamma = float(global_parameters["gamma"])
            except Exception:
                self.gamma = str(global_parameters["gamma"])  # 'scale' or 'auto'
        if "degree" in global_parameters:
            try:
                self.degree = int(global_parameters["degree"])
            except Exception:
                pass
        if "coef0" in global_parameters:
            self.coef0 = float(global_parameters["coef0"])

        if self.kernel == "linear":
            if "weights" in global_parameters and "biases" in global_parameters:
                self.weights = np.array(global_parameters["weights"]).ravel()
                self.biases = float(np.array(global_parameters["biases"]).flatten()[0])
                # Switch to manual path when loading raw params
                self.use_sklearn = False
        else:
            if (
                "support_vectors" in global_parameters
                and "dual_coef" in global_parameters
                and "intercept" in global_parameters
            ):
                self.support_vectors = np.array(global_parameters["support_vectors"])
                self.dual_coef = np.array(global_parameters["dual_coef"]).ravel()
                self.biases = float(
                    np.array(global_parameters["intercept"]).flatten()[0]
                )
                if "gamma_resolved" in global_parameters:
                    try:
                        self.gamma_resolved = float(global_parameters["gamma_resolved"])
                    except Exception:
                        self.gamma_resolved = None
                if (
                    "scaler_mean" in global_parameters
                    and "scaler_scale" in global_parameters
                ):
                    try:
                        self.scaler = StandardScaler()
                        self.scaler.mean_ = np.array(
                            global_parameters["scaler_mean"]
                        ).ravel()
                        self.scaler.scale_ = np.array(
                            global_parameters["scaler_scale"]
                        ).ravel()
                        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]
                    except Exception:
                        self.scaler = None
                self.use_sklearn = False

    def get_parameters(self):
        if self.use_sklearn and self.sklearn_model is not None:
            w = None
            b = None
            if self.kernel == "linear":
                try:
                    w = np.array(self.sklearn_model.coef_).ravel().tolist()
                except Exception:
                    w = self.weights.tolist() if self.weights is not None else None
                try:
                    b = float(np.array(self.sklearn_model.intercept_).flatten()[0])
                except Exception:
                    b = float(self.biases) if self.biases is not None else None
            if self.kernel != "linear":
                try:
                    sv = np.array(self.sklearn_model.support_vectors_).tolist()
                    dc = np.array(self.sklearn_model.dual_coef_).ravel().tolist()
                    b = float(np.array(self.sklearn_model.intercept_).flatten()[0])
                    scaler_mean = (
                        self.scaler.mean_.tolist()
                        if hasattr(self.scaler, "mean_")
                        else None
                    )
                    scaler_scale = (
                        self.scaler.scale_.tolist()
                        if hasattr(self.scaler, "scale_")
                        else None
                    )
                except Exception:
                    sv = None
                    dc = None
                    scaler_mean = None
                    scaler_scale = None
                return {
                    "support_vectors": sv,
                    "dual_coef": dc,
                    "intercept": b,
                    "scaler_mean": scaler_mean,
                    "scaler_scale": scaler_scale,
                }
            return {"weights": w, "biases": b}

        if self.kernel == "linear":
            if self.weights is None or self.biases is None:
                return {"weights": None, "biases": None}
            return {
                "weights": np.array(self.weights).ravel().tolist(),
                "biases": float(np.array(self.biases).flatten()[0]),
            }
        sv = self.support_vectors.tolist() if self.support_vectors is not None else None
        dc = self.dual_coef.tolist() if self.dual_coef is not None else None
        scaler_mean = (
            self.scaler.mean_.tolist()
            if (self.scaler is not None and hasattr(self.scaler, "mean_"))
            else None
        )
        scaler_scale = (
            self.scaler.scale_.tolist()
            if (self.scaler is not None and hasattr(self.scaler, "scale_"))
            else None
        )
        return {
            "support_vectors": sv,
            "dual_coef": dc,
            "intercept": (
                float(np.array(self.biases).flatten()[0])
                if self.biases is not None
                else None
            ),
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
        }

    # --------------------------
    # Evaluation
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
            elif name == "r2" or name == "r2_score":
                # Compute coefficient of determination
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                results["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        return results
