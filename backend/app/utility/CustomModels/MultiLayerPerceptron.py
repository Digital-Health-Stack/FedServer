import numpy as np



# model_params is dictionary
# model_params['input_layer']: Also a dict num_nodes,activation_function,input_shape
# model_params['intermediate_layer']: List of dict num_nodes,activation_function
# model_params['output_layer']: Also a dict num_nodes,activation_function
# model_params['loss']
# model_params['optimizer']
class MultiLayerPerceptron:
    def __init__(self, model_params,epochs=1, lr=0.01):
        from keras import Sequential
        from keras.layers import Dense
        self.lr = lr
        self.epochs = epochs

        # Intitialising the model
        self.model = Sequential()
        
        # Input Layer
        # Input Shape parameter in format of (124,)
        self.model.add(Dense(model_params['input_layer']['num_nodes'], activation=model_params['input_layer']['activation_function'],input_shape=model_params['input_layer']['input_shape']))

        # Intermediate Layer
        for i in range(len(model_params['intermediate_layer'])):
            self.model.add(Dense(model_params['intermediate_layer'][i]['num_nodes'], activation=model_params['intermediate_layer'][i]['activation_function']))

        #Output Layer
        self.model.add(Dense(model_params['output_layer']['num_nodes']),activation=model_params['output_layer']['activation_function'])
        
        # Model Configuration
        self.model.compile(loss=model_params['loss'], optimizer=model_params['optimizer'], metrics=['accuracy'])
        self.history = None
        print("Model Summary : ", self.model.summary())

    def fit(self, X_train, Y_train):
        # Ensure X_train and Y_train are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        # Get the number of samples
        self.history = self.model.fit(X_train, Y_train, epochs=self.epochs, validation_split=0.2)

    def predict(self, X):
        y_prob =  self.model.predict(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred

    def get_loss_validation(self):
        metric = [self.history.history['loss'], self.history.history['val_loss'], self.history.history['accuracy'],self.history.history['val_accuracy']]

        return metric

    def update_parameters(self, parameters_dict):
        if len(parameters_dict) == 0:
            return
        self.model.set_weights(parameters_dict)

    def get_parameters(self):
        parameters = self.model.get_weights()
        return parameters

    def change_model_parameters(self, client_iter, client_learning_rate):
        self.n_iters = client_iter
        self.client_learning_rate = client_learning_rate
