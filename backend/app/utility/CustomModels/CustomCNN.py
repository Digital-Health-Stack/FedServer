import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import ast




def handle_error(error):
    error_message = f"An error occurred: {error}"
    print(error_message)


class CustomCNN:
    def __init__(self, config):
        from tensorflow.keras import regularizers, optimizers   
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Input, Dense, Flatten, Conv2D, Reshape, 
                                    MaxPooling2D, AveragePooling2D, BatchNormalization,
                                    Dropout)
        self.config = config
        self.model = Sequential()
        try:
            input_shape = ast.literal_eval(config.get('input_shape', '(128, 128, 1)'))
            self.model.add(Input(shape=input_shape)) 
            
            # Adding layers to the model
            for i in range(len(config['layers'])):
                layer = config['layers'][i]
                layer_type = layer['layer_type']

                if layer_type == 'dense':
                    dense_layer = Dense(
                        int(layer['num_nodes']), 
                        activation=layer.get('activation_function', 'relu')
                    )
                    # Add regularization if specified
                    if 'regularizer' in layer:
                        reg_config = layer['regularizer']
                        if reg_config['type'] == 'l2':
                            dense_layer.kernel_regularizer = regularizers.l2(float(reg_config['factor']))
                    self.model.add(dense_layer)
                    
                elif layer_type == 'flatten':
                    self.model.add(Flatten())
                    
                elif layer_type == 'convolution':
                    conv_layer = Conv2D(
                        int(layer['filters']),
                        kernel_size=eval(layer.get('kernel_size', '(3, 3)')),
                        strides=eval(layer.get('stride', '(1, 1)')),
                        activation=layer.get('activation_function', 'relu'),
                        padding=layer.get('padding', 'same')
                    )
                    self.model.add(conv_layer)
                    
                elif layer_type == 'reshape':
                    self.model.add(Reshape(eval(layer['target_shape'])))
                    
                elif layer_type == 'pooling':
                    if layer.get("pooling_type", "max") == "max":
                        self.model.add(MaxPooling2D(
                            pool_size=eval(layer.get('pool_size', '(2, 2)')), 
                            strides=eval(layer.get('stride', 'None'))  # None means same as pool_size
                        ))
                    elif layer.get("pooling_type") == "average":
                        self.model.add(AveragePooling2D(
                            pool_size=eval(layer.get('pool_size', '(2, 2)')), 
                            strides=eval(layer.get('stride', 'None'))
                        ))
                        
                elif layer_type == 'batch_norm':
                    self.model.add(BatchNormalization())
                    
                elif layer_type == 'dropout':
                    self.model.add(Dropout(float(layer.get('rate', 0.5))))
            
            # Output layer
            output_layer = config['output_layer']
            output_dense = Dense(
                int(output_layer['num_nodes']), 
                activation=output_layer.get('activation_function', 'linear')
            )
            if 'regularizer' in output_layer:
                reg_config = output_layer['regularizer']
                if reg_config['type'] == 'l2':
                    output_dense.kernel_regularizer = regularizers.l2(float(reg_config['factor']))
            self.model.add(output_dense)

            # Compile model
            optimizer_config = config.get('optimizer', {'type': 'adam', 'learning_rate': 0.001})
            print("Checkpoint 1 : ", optimizer_config)
            if optimizer_config['type'] == 'adam':
                optimizer = optimizers.Adam(learning_rate=float(optimizer_config.get('learning_rate', 0.001)))
            elif optimizer_config['type'] == 'sgd':
                optimizer = optimizers.SGD(learning_rate=float(optimizer_config.get('learning_rate', 0.01)))
            else:
                optimizer = optimizer_config['type']  # assume it's a string like 'adam'
                
            metrics_config = config.get('metrics', {})
            metrics = [m for m, enabled in metrics_config.items() if enabled] if isinstance(metrics_config, dict) else (
                [metrics_config] if isinstance(metrics_config, str) else metrics_config
            )
            if not metrics:  # Default if empty
                metrics = ['mae']
                
            self.model.compile(
                loss=config.get('loss', 'mean_squared_error'),
                optimizer=optimizer,
                metrics=metrics
            )

        except Exception as e:
            handle_error(e)

    def fit(self, X, y, epochs=1, batch_size=32, validation_data=None, callbacks=None):
        try:
            # Input validation
            if X is None or y is None:
                print("Error: X and y cannot be None")
                return None
            
            # Check model compilation
            if not hasattr(self.model, 'optimizer'):
                print("Error: Model is not compiled")
                return None
            return self.model.fit(
                X, y, 
                epochs=epochs, 
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks
            )
        except Exception as e:
            error_message = f"An error occurred in Fit Function: {e}"
            print(error_message)

    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            handle_error(e)

    def get_parameters(self):
        try:
            params = {'weights': []}
            for i, layer in enumerate(self.model.layers):
                layer_weights = layer.get_weights()
                # Convert each weight array to lists (can't do at once.)
                params['weights'].append([w.tolist() for w in layer_weights])
            return params
        except Exception as e:
            handle_error(e)

    def update_parameters(self, new_params):
        try:
            for i, (layer, layer_params) in enumerate(zip(self.model.layers, new_params['weights'])):
                # Convert lists back to numpy arrays
                layer.set_weights([np.array(w) for w in layer_params])
        except Exception as e:
            handle_error(e)