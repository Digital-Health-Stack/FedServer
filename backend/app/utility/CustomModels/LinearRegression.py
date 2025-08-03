import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self, config):
        self.lr = float(config.get("lr", 0.01))
        self.n_iters = int(config.get("n_iters", 100))
        self.m = None
        self.c = None

    def fit(self, X_train, Y_train):
        # Ensure X_train and Y_train are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Get the number of samples
        num_samples, num_features = X_train.shape

        # Initialize parameters
        if self.m is None:
            self.m = np.zeros(num_features)  # Initialize m with zeros
        if self.c is None:
            self.c = 0  # Initialize c with zero

        # Gradient Descent
        for i in range(self.n_iters):
            Y_pred = np.dot(X_train, self.m) + self.c
            # Ensure Y_train and Y_pred have the same shape
            Y_train_reshaped = Y_train.reshape(-1)
            Y_pred_reshaped = Y_pred.reshape(-1)

            residuals = Y_train_reshaped - Y_pred_reshaped
            # Calculate gradients
            D_m = (-2 / num_samples) * np.dot(X_train.T, residuals)
            D_c = (-2 / num_samples) * np.sum(residuals)

            self.m = self.m - self.lr * D_m
            self.c = self.c - self.lr * D_c

    def predict(self, X):
        X = np.array(X)
        pred_test = np.dot(X, self.m) + self.c
        return pred_test

    def update_parameters(self, parameters_dict):
        # Convert 'm' from list to numpy array if it's not None
        self.m = (
            np.array(parameters_dict["m"]) if parameters_dict["m"] is not None else None
        )
        # Convert 'c' from list to float if it's not None
        self.c = (
            float(parameters_dict["c"][0]) if parameters_dict["c"] is not None else None
        )

    def get_parameters(self):
        # Convert 'm' to list if it's not None
        # Wrap 'c' in a list for sending if it's not None
        local_parameter = {
            "m": self.m.tolist() if self.m is not None else None,
            "c": [self.c] if self.c is not None else None,
        }
        return local_parameter

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return {"mse": mean_squared_error(Y_pred, Y_test)}
