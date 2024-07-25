import numpy as np
import warnings

"""
(i) This implementation of the Support Vector Machine (SVM) algorithm implements Linear SVM (both binary and Multi-class)
 and Landmark-based Kernel SVM.
 the dual form kernel trick has not been implemented...because in the federated setting we can't share 
 the support vectors to the server, which are the key to the kernel trick...although the kernel trick can be implemented 
 in a secure multi-party computation setting but this problem has some other obvious challenges.

(ii) If a new client wants to use this class to update the pretrained model (parameters on some server),
 it is always recommended to first fetch the parameters from the server hae a idea about dimension
  of parameter then define your custom model with the same dimensions to not have any dimension mismatch.

(iii) landmarks arg in the constructor is only used when is_landmark_based is set to True,
 if given it should be any iterable with the points having the same dimension as the input data.
 if not given random 'num_landmarks' number of points will be selected from the input data.

 (iv) if landmarks are given, num_landmarks will not be used (these are used to select random points from the input data)

(v) The dimension of weights on the server must be same as the weights of the model to be updated/send to the server.

(vi) landmarks should be num_landmarks datapoints from the input data, if not given random points will be selected from the input data.
"""

def transform_by_landmarks(X, kernel, gamma=None, degree=None, coef0=None, landmarks=None, num_landmarks=15):
    if landmarks is None:
        landmarks = X[np.random.choice(X.shape[0], num_landmarks, replace=True)]

    if kernel == 'rbf':
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - landmarks, axis=2)
        transformed_X = np.exp(-gamma * (dist_matrix ** 2))

    elif kernel == 'linear':
        transformed_X = X @ landmarks.T

    elif kernel == 'polynomial':
        transformed_X = (X @ landmarks.T + coef0) ** degree

    elif kernel == 'sigmoid':
        transformed_X = np.tanh(gamma * (X @ landmarks.T) + coef0)

    else:
        raise ValueError("Invalid kernel function.")

    return transformed_X


class LandMarkSVM:
    def __init__(self, C=1.0, is_binary=False, kernel='rbf', gamma='auto', degree=3, coef0=0.0, lr=0.01, n_iters=100,
                 landmarks=None, num_landmarks=15, weights_shape=None):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.weights = None
        self.biases = None
        self.lr = lr
        self.n_iters = n_iters
        self.is_binary = is_binary
        self.kernel = kernel
        self.landmarks = landmarks
        self.num_landmarks = num_landmarks

        if weights_shape is not None:
            self.weights = np.zeros(weights_shape)
            self.biases = np.zeros(weights_shape[0])

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
        X = transform_by_landmarks(X, self.kernel, self.gamma, self.degree, self.coef0, self.landmarks, self.num_landmarks)

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
        X = transform_by_landmarks(X, self.kernel, self.gamma, self.degree, self.coef0, self.landmarks, self.num_landmarks)
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
            raise ValueError("Parameters are None")
        local_parameter = {'weights': self.weights.tolist(), 'biases': self.biases.tolist()}
        return local_parameter
    
    def save_model(self, file_path):
        np.savez(file_path, weights=self.weights, biases=self.biases)


    def load_model(self, file_path):
        try:
            data = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found.")

        self.weights = data['weights']
        self.biases = data['biases']
        return self.weights, self.biases
    #  ==================================================================================================
    # The following methods are not used in the federated setting





    def change_n_iters(self, client_iter):
        self.n_iters = client_iter

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return self.weights

    def get_biases(self):
        if self.biases is None:
            raise ValueError("Model has not been trained yet.")
        return self.biases

    def update_weights(self, new_weights):
        """ new weights should be a numpy array """
        self.weights = new_weights

    def update_biases(self, new_biases):
        """ new biases should be a numpy array """
        self.biases = new_biases
