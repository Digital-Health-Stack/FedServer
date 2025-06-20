import numpy as np
import warnings
import ast

"""
(i) read CustomSVM documentation then go through this 

(iii) landmarks arg in the constructor is only used when is_landmark_based is set to True,
 if given it should be any iterable with the points having the same dimension as the input data.
 if not given random 'num_landmarks' number of points will be selected from the input data.

(iv) if landmarks are given, num_landmarks will not be used (these are used to select random points from the input data)

(v) The dimension of weights on the server must be same as the weights of the model to be updated/send to the server.

(vi) landmarks should be num_landmarks datapoints from the input data, if not given random points will be selected from the input data.
"""


def transform_by_landmarks(
    X, kernel, gamma=None, degree=None, coef0=None, landmarks=None, num_landmarks=15
):
    if landmarks is None:
        landmarks = X[np.random.choice(X.shape[0], num_landmarks, replace=True)]

    if kernel == "rbf":
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - landmarks, axis=2)
        transformed_X = np.exp(-gamma * (dist_matrix**2))

    elif kernel == "linear":
        transformed_X = X @ landmarks.T

    elif kernel == "polynomial":
        transformed_X = (X @ landmarks.T + coef0) ** degree

    elif kernel == "sigmoid":
        transformed_X = np.tanh(gamma * (X @ landmarks.T) + coef0)

    else:
        raise ValueError("Invalid kernel function.")

    return transformed_X


class LandMarkSVM:
    def __init__(self, config):
        try:
            # Extract and convert parameters from the config dictionary
            self.C = float(config.get("C", 1.0))
            self.gamma = config.get("gamma", "auto")
            self.degree = int(config.get("degree", 3))
            self.coef0 = float(config.get("coef0", 0.0))
            self.lr = float(config.get("lr", 0.01))
            self.n_iters = int(config.get("n_iters", 100))
            self.is_binary = config.get("is_binary", "false").lower() == "true"
            self.kernel = config.get("kernel", "rbf")
            self.landmarks = config.get("landmarks", None)
            self.num_landmarks = int(config.get("num_landmarks", 15))

            # Handle weights_shape safely
            weights_shape_str = config.get("weights_shape", None)
            if weights_shape_str is not None:
                self.weights_shape = ast.literal_eval(weights_shape_str)
                self.weights = np.zeros(self.weights_shape)
                self.biases = np.zeros(self.weights_shape[0])
            else:
                self.weights_shape = None
                self.weights = None
                self.biases = None

        except Exception as e:
            print(f"Error creating model instance: {e}")
            self.weights = None
            self.biases = None

    def fit_binary(self, X, y):
        self.is_binary = True
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        lr = self.lr
        n_iters = self.n_iters
        binary_y = np.where(y == 1, 1, -1)
        for epoch in range(n_iters):
            for idx, x in enumerate(X):
                decision = np.dot(x, weights) + bias
                if binary_y[idx] * decision < 1:
                    weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                    bias += lr * binary_y[idx]
                else:
                    weights += lr * (-2 * self.C * weights)
        self.weights = weights
        self.biases = bias

    def fit(self, X, y):
        # if self.weights is not None or self.biases is not None:
        #     print("weight before fit:", self.get_weights().tolist())
        # else:
        #     print("weight before fit: None")
        X = transform_by_landmarks(
            X,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
            self.landmarks,
            self.num_landmarks,
        )

        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(np.unique(y))
        if self.weights is None and (n_classes == 2 or self.is_binary):
            self.fit_binary(X, y)
            return

        if self.weights is None and self.biases is None:
            self.weights = np.zeros((n_classes, n_features))
            self.biases = np.zeros(n_classes)
        # print("weights shape", self.weights.shape)
        for i in classes:
            i = int(i)
            binary_y = np.where(y == i, 1, -1)
            weights = self.weights[i]
            bias = self.biases[i]
            lr = self.lr  # Learning rate
            n_iters = self.n_iters  # Number of n_iters
            for epoch in range(n_iters):
                for idx, x in enumerate(X):
                    decision = np.dot(x, weights) + bias
                    if binary_y[idx] * decision < 1:
                        weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                        bias += lr * binary_y[idx]
                    else:
                        weights += lr * (-2 * self.C * weights)
            self.weights[i] = weights
            self.biases[i] = bias
        # print("weight after fit:", self.get_weights().tolist())

    def predict(self, X):
        if self.weights is None or self.biases is None:
            warnings.warn("Model has not been trained yet...NONE weights and biases.")
            return np.array([0])
        X = transform_by_landmarks(
            X,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
            self.landmarks,
            self.num_landmarks,
        )
        decision_values = np.dot(X, self.weights.T) + self.biases

        if self.is_binary:
            return np.where(decision_values < 0, 0, 1)
        return np.argmax(decision_values, axis=1)

    def update_parameters(self, parameters):
        """parameters should be a dictionary with keys 'weights' and 'biases' where values are lists"""
        self.weights = np.array(parameters["weights"])
        self.biases = np.array(parameters["biases"])

    def get_parameters(self):
        if self.weights is None and self.biases is None:
            raise ValueError("Parameters are None")
        local_parameter = {
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }
        return local_parameter
