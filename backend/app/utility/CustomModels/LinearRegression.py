import numpy as np
# from sklearn.preprocessing import StandardScaler


'''
    Assumptions: 
    1. Datatype is int or float
    2. Features are scaled using StandardScaler
    3. Missing Values are handled
'''
class LinearRegression:
    def __init__(self , config):
        self.lr = float(config.get('lr', 0.01))
        self.n_iters = int(config.get('n_iters', 100))
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
            # print("Check kar : ",Y_pred[:3],Y_train[:3])
            residuals = Y_train - Y_pred
            # print("m :",self.m[:5])
            # print(X_train.shape,residuals.shape)
            # print(np.dot(X_train.T,residuals))
            #print("c" , self.c[:5])
            #print("Check kar : ",X_train[:5])
            D_m = (-2 / num_samples) * np.dot(X_train.T, residuals)
            D_c = (-2 / num_samples) * np.sum(residuals)
            
            # print("Check kar L ",D_m,D_c)
            self.m = self.m - self.lr * D_m
            self.c = self.c - self.lr * D_c

    def predict(self, X):
        X = np.array(X)
        pred_test = np.dot(X,self.m) + self.c
        return pred_test
    
    def update_parameters(self, parameters_dict):
        # Convert 'm' from list to numpy array if it's not None
        self.m = np.array(parameters_dict['m']) if parameters_dict['m'] is not None else None
        # Convert 'c' from list to float if it's not None
        self.c = float(parameters_dict['c'][0]) if parameters_dict['c'] is not None else None

    def get_parameters(self):
        # Convert 'm' to list if it's not None
        # Wrap 'c' in a list for sending if it's not None
        local_parameter = {'m': self.m.tolist() if self.m is not None else None, 
                       'c': [self.c] if self.c is not None else None}
        return local_parameter
    
    def change_n_iters(self,client_iter):
        self.n_iters = client_iter
