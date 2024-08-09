import numpy as np
import warnings
import ast

"""
(i) This implementation of the Support Vector Machine (SVM) algorithm implements Linear SVM (both binary and Multi-class)
 the dual form kernel trick has not been implemented...because in the federated setting we can't share 
 the support vectors to the server, which are the key to the kernel trick...although the kernel trick can be implemented 
 in a secure multi-party computation setting but this problem has some other obvious challenges.

(ii) If a new client wants to use this class to update the pretrained model (parameters on some server),
 it is always recommended to first fetch the parameters from the server has a idea about dimension
  of parameter then define your custom model with the same dimensions to not have any dimension mismatch.

(iii) landmarks arg in the constructor is only used when is_landmark_based is set to True,
 if given it should be any iterable with the points having the same dimension as the input data.
 if not given random 'num_landmarks' number of points will be selected from the input data.

(iv) if landmarks are given, num_landmarks will not be used (these are used to select random points from the input data)

(v) The dimension of weights on the server must be same as the weights of the model to be updated/send to the server.
"""


class CustomSVM:
    def __init__(self, config ):
        try:
            self.C = float(config.get('C', 1.0))
            self.weights = None
            self.biases = None
            self.lr = float(config.get('lr', 0.01))
            self.n_iters = int(config.get('n_iters', 100))
            self.weights_shape = ast.literal_eval(config['weights_shape'])
            #required as each client may have different number of classes (possibly 1)
            self.is_binary = config['is_binary'].lower() == "true"
            # weight shape is required initially for the same reason as above
            if self.weights_shape is not None:
                self.weights = np.zeros(self.weights_shape)
                self.biases = np.zeros(self.weights_shape)
        except Exception as e:
            print(f"Error creating model instance: {e}")
            return None

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
        for i in classes: #*** 'i' can be float due to numpy
            i = int(i)
            binary_y = np.where(y == i, 1, -1)
            weights = self.weights[i]
            bias = self.biases[i]
            lr = self.lr  # Learning rate
            n_iters = self.n_iters  # Number of n_iters
            for _ in range(n_iters):
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
        decision_values = np.dot(X, self.weights.T) + self.biases

        if self.is_binary:
            return np.where(decision_values < 0, 0, 1)
        return np.argmax(decision_values, axis=1)

    def update_parameters(self, parameters):
        """ parameters should be a dictionary with keys 'weights' and 'biases' where values are lists """
        self.weights = np.array(parameters['weights'])
        self.biases = np.array(parameters['biases'])

    def get_parameters(self):
        if self.weights is None and self.biases is None:
            raise ValueError("Local Parameters are None")
        local_parameter = {'weights': self.weights.tolist(), 'biases': self.biases.tolist()}
        return local_parameter


