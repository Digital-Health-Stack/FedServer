import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


class DecisionTree:
    def __init__(
        self,
        config=None,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        task_type="classification",
    ):
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

        def _to_str(value, default):
            try:
                return str(value)
            except Exception:
                return str(default)

        if isinstance(config, dict):
            self.max_depth = _to_int(config.get("max_depth", max_depth), max_depth)
            self.min_samples_split = _to_int(
                config.get("min_samples_split", min_samples_split), min_samples_split
            )
            self.min_samples_leaf = _to_int(
                config.get("min_samples_leaf", min_samples_leaf), min_samples_leaf
            )
            self.task_type = _to_str(config.get("task_type", task_type), task_type)
        else:
            self.max_depth = _to_int(max_depth, 5)
            self.min_samples_split = _to_int(min_samples_split, 2)
            self.min_samples_leaf = _to_int(min_samples_leaf, 1)
            self.task_type = _to_str(task_type, "classification")

        self.sklearn_model = None
        self.x_scaler = None
        self.feature_importances_ = None
        self.tree_structure = None

    # --------------------------
    # FIT METHOD
    # --------------------------
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        try:
            self.fit_sklearn(X, y)
        except Exception as e:
            print(f"Error fitting Decision Tree: {e}")
            raise

    def fit_sklearn(self, X, y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        # Optional scaling for consistency with other models
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)

        # Choose classifier or regressor based on task type
        if self.task_type == "classification":
            self.sklearn_model = DecisionTreeClassifier(
                max_depth=self.max_depth if self.max_depth > 0 else None,
                min_samples_split=max(2, self.min_samples_split),
                min_samples_leaf=max(1, self.min_samples_leaf),
                random_state=42,
            )
        else:  # regression
            self.sklearn_model = DecisionTreeRegressor(
                max_depth=self.max_depth if self.max_depth > 0 else None,
                min_samples_split=max(2, self.min_samples_split),
                min_samples_leaf=max(1, self.min_samples_leaf),
                random_state=42,
            )

        self.sklearn_model.fit(X_scaled, y)
        self.feature_importances_ = self.sklearn_model.feature_importances_

        # Store simplified tree structure for parameter sharing
        self._extract_tree_structure()

    def _extract_tree_structure(self):
        """Extract simplified tree structure for federated parameter sharing"""
        if self.sklearn_model is None:
            return

        tree = self.sklearn_model.tree_
        self.tree_structure = {
            "feature_importances": [
                float(x) for x in self.feature_importances_.tolist()
            ],
            "n_features": int(tree.n_features),
            "n_classes": int(
                getattr(tree, "n_classes", [1])[0] if hasattr(tree, "n_classes") else 1
            ),
            "max_depth": int(self.sklearn_model.get_depth()),
            "n_leaves": int(tree.n_leaves),
            "n_nodes": int(tree.node_count),
        }

    # --------------------------
    # HELPER METHOD FOR INPUT PREPARATION
    # --------------------------
    def _prepare_input(self, X):
        """Prepare input data for prediction, handling scaler availability"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Use scaler if available, otherwise use raw data
        if self.x_scaler is not None:
            return self.x_scaler.transform(X)
        else:
            return X

    # --------------------------
    # PREDICT METHOD
    # --------------------------
    def predict(self, X):
        X_processed = self._prepare_input(X)

        if self.sklearn_model is None:
            # Fallback prediction when no trained model is available
            return self._fallback_predict(X_processed)

        if self.task_type == "classification":
            # Return probabilities for class 1 for binary classification
            if hasattr(self.sklearn_model, "predict_proba"):
                proba = self.sklearn_model.predict_proba(X_processed)
                if proba.shape[1] == 2:
                    return proba[:, 1]
                else:
                    # Multi-class: return the max probability
                    return np.max(proba, axis=1)
            else:
                return self.sklearn_model.predict(X_processed).astype(float)
        else:
            return self.sklearn_model.predict(X_processed)

    def _fallback_predict(self, X):
        """Provide fallback predictions when sklearn_model is not available"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # If we have feature importances, use them for a simple prediction
        if self.feature_importances_ is not None:
            feature_importances = np.array(self.feature_importances_)

            # Ensure feature dimensions match
            if X.shape[1] == len(feature_importances):
                # Simple weighted sum based on feature importances
                weighted_scores = np.dot(X, feature_importances)

                if self.task_type == "classification":
                    # For classification, return probabilities based on normalized scores
                    # Normalize scores to [0, 1] range
                    min_score, max_score = weighted_scores.min(), weighted_scores.max()
                    if max_score > min_score:
                        normalized_scores = (weighted_scores - min_score) / (
                            max_score - min_score
                        )
                    else:
                        normalized_scores = np.full(weighted_scores.shape, 0.5)
                    return normalized_scores
                else:
                    # For regression, return the weighted scores directly
                    return weighted_scores

        # Ultimate fallback: return neutral predictions
        if self.task_type == "classification":
            # Return 0.5 probability for binary classification
            return np.full(X.shape[0], 0.5)
        else:
            # Return zeros for regression
            return np.zeros(X.shape[0])

    # --------------------------
    # PREDICT CLASSES
    # --------------------------
    def predict_classes(self, X, threshold=0.5):
        """Predict classes for classification tasks"""
        if self.task_type == "classification":
            # Get probabilities from predict method (handles sklearn_model=None case)
            probabilities = self.predict(X)

            if self.sklearn_model is None:
                # Fallback: use threshold-based classification for binary, argmax for multi-class
                if self.tree_structure and self.tree_structure.get("n_classes", 2) == 2:
                    return (probabilities >= threshold).astype(int)
                else:
                    # For multi-class without trained model, return random class
                    n_classes = (
                        self.tree_structure.get("n_classes", 2)
                        if self.tree_structure
                        else 2
                    )
                    return np.random.randint(0, n_classes, size=len(probabilities))

            # Standard sklearn model path
            if hasattr(self.sklearn_model, "predict_proba"):
                if self.sklearn_model.classes_.size == 2:
                    return (probabilities >= threshold).astype(int)
                else:
                    # Multi-class: use argmax
                    X_processed = self._prepare_input(X)
                    return self.sklearn_model.predict(X_processed)
            else:
                X_processed = self._prepare_input(X)
                return self.sklearn_model.predict(X_processed)
        else:
            return self.predict(X)

    # --------------------------
    # UPDATE PARAMETERS
    # --------------------------
    def update_parameters(self, global_parameters):
        """Update model parameters - for Decision Trees, this mainly updates hyperparameters"""
        if global_parameters is not None:
            # For federated learning, we can share feature importances and tree statistics
            if "feature_importances" in global_parameters:
                self.feature_importances_ = np.array(
                    global_parameters["feature_importances"]
                )

            if "tree_structure" in global_parameters:
                self.tree_structure = global_parameters["tree_structure"]

                # Initialize sklearn_model if it doesn't exist and we have enough information
                if self.sklearn_model is None and isinstance(self.tree_structure, dict):
                    self._initialize_sklearn_model_from_structure()

    def _initialize_sklearn_model_from_structure(self):
        """Initialize a basic sklearn model when loading from federated parameters"""
        try:
            # Create a basic model instance for prediction purposes
            if self.task_type == "classification":
                self.sklearn_model = DecisionTreeClassifier(
                    max_depth=self.max_depth if self.max_depth > 0 else None,
                    min_samples_split=max(2, self.min_samples_split),
                    min_samples_leaf=max(1, self.min_samples_leaf),
                    random_state=42,
                )
            else:  # regression
                self.sklearn_model = DecisionTreeRegressor(
                    max_depth=self.max_depth if self.max_depth > 0 else None,
                    min_samples_split=max(2, self.min_samples_split),
                    min_samples_leaf=max(1, self.min_samples_leaf),
                    random_state=42,
                )

            # Get n_features and n_classes from tree_structure if available
            n_features = self.tree_structure.get("n_features", 1)
            n_classes = self.tree_structure.get("n_classes", 2)

            # Create dummy data to initialize the model structure
            if self.task_type == "classification":
                # Create dummy classification data
                X_dummy = np.random.random((max(n_classes, 10), n_features))
                y_dummy = np.random.randint(0, n_classes, max(n_classes, 10))
            else:
                # Create dummy regression data
                X_dummy = np.random.random((10, n_features))
                y_dummy = np.random.random(10)

            # Fit the model on dummy data to establish the structure
            self.sklearn_model.fit(X_dummy, y_dummy)

            # Initialize scaler for consistency
            if self.x_scaler is None:
                self.x_scaler = StandardScaler()
                self.x_scaler.fit(X_dummy)

        except Exception as e:
            print(f"Warning: Could not initialize sklearn model from structure: {e}")
            # Set sklearn_model to None to use fallback prediction
            self.sklearn_model = None

    # --------------------------
    # GET PARAMETERS
    # --------------------------
    def get_parameters(self):
        """Get model parameters for federated sharing"""

        def _safe_float(value):
            """Convert value to float, replacing NaN/inf with 0.0"""
            try:
                val = float(value)
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            except (ValueError, TypeError):
                return 0.0

        def _safe_float_list(arr):
            """Convert array to list of safe floats"""
            try:
                arr = np.array(arr).ravel()
                return [_safe_float(v) for v in arr.tolist()]
            except:
                return [0.0]

        def _safe_int(value):
            """Convert value to int, handling numpy types"""
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0

        parameters = {}

        # Add model-specific parameters if available
        if self.feature_importances_ is not None:
            parameters["feature_importances"] = _safe_float_list(
                self.feature_importances_
            )

        if self.tree_structure is not None:
            parameters["tree_structure"] = self.tree_structure

        return parameters

    # --------------------------
    # EVALUATE METHOD
    # --------------------------
    def evaluate(self, X, y, metrics):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        print("Metrics: ", metrics)

        results = {}

        if self.task_type == "classification":
            y_prob = self.predict(X)
            y_pred = self.predict_classes(X)

            for metric in metrics or []:
                name = metric.lower()
                if name == "accuracy":
                    results["accuracy"] = float(np.mean(y == y_pred))
                elif name == "precision":
                    tp = np.sum((y == 1) & (y_pred == 1))
                    fp = np.sum((y == 0) & (y_pred == 1))
                    results["precision"] = (
                        float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    )
                elif name == "recall":
                    tp = np.sum((y == 1) & (y_pred == 1))
                    fn = np.sum((y == 1) & (y_pred == 0))
                    results["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                elif name == "f1_score" or name == "f1":
                    tp = np.sum((y == 1) & (y_pred == 1))
                    fp = np.sum((y == 0) & (y_pred == 1))
                    fn = np.sum((y == 1) & (y_pred == 0))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    results["f1_score"] = (
                        float(2 * precision * recall / (precision + recall))
                        if (precision + recall) > 0
                        else 0.0
                    )
                elif name == "log_loss":
                    # For classification with probabilities
                    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                    results["log_loss"] = float(
                        -np.mean(
                            y * np.log(y_prob_clipped)
                            + (1 - y) * np.log(1 - y_prob_clipped)
                        )
                    )
        else:  # regression
            y_pred = self.predict(X)

            for metric in metrics or []:
                name = metric.lower()
                if name == "mse" or name == "mean_squared_error":
                    results["mse"] = float(np.mean((y - y_pred) ** 2))
                elif name == "mae" or name == "mean_absolute_error":
                    results["mae"] = float(np.mean(np.abs(y - y_pred)))
                elif name == "r2" or name == "r2_score":
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    results["r2"] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
                elif name == "rmse" or name == "root_mean_squared_error":
                    results["rmse"] = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        return results

    # --------------------------
    # GET FEATURE IMPORTANCES
    # --------------------------
    def get_feature_importances(self):
        """Get feature importances from the trained tree"""
        if self.feature_importances_ is not None:
            return self.feature_importances_.tolist()
        return None
