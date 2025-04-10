import ast
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, MaxPooling2D, AveragePooling2D
from elephas.spark_model import SparkModel
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanAbsoluteError
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomCNN:
    def __init__(self, config, mode='asynchronous', num_workers=2):
        """
        Initialize CNN model with error handling and GPU support detection.
        """
        self.config = config
        # self.spark_context = spark_context  # This is problematic for pickling serialize
        self.model = None
        self.spark_model = None

        try:
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logging.info("GPU detected. Configuring TensorFlow to use GPU.")
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                except RuntimeError as e:
                    logging.error(f"GPU configuration error: {e}")
            else:
                logging.info("No GPU detected. Using CPU for training.")

            # Build model
            self.model = Sequential()
            input_shape = ast.literal_eval(config['model_info']['input_shape'])
            self.model.add(Input(shape=input_shape))

            # Add layers from config
            for layer in config['model_info']['layers']:
                self._add_layer(layer)

            # Add output layer
            output_layer = config['model_info']['output_layer']
            self.model.add(Dense(
                int(output_layer['num_nodes']),
                activation=output_layer['activation_function']
            ))

            sgd = SGD(lr=0.1)
            # Compile with metrics from config
            self.model.compile(sgd, 'categorical_crossentropy',['acc'] )  

            # return self.model

            # Initialize Spark model
            self.spark_model = SparkModel(
                self.model,
                mode=mode,
                num_workers=num_workers,
                frequency='epoch',
                # custom_objects=self._get_custom_objects()
            )
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    # def _get_custom_objects(self):
    #     """Ensure custom objects are available for Spark serialization"""
    #     return {
    #         'Adam': tf.keras.optimizers.Adam,
    #         'MSE': tf.keras.losses.MeanSquaredError,
    #         'mae': tf.keras.metrics.MeanAbsoluteError  # Must Match the metric name used in compile
    #     }

    def _add_layer(self, layer_config):
        """Helper to add layers based on configuration"""
        layer_type = layer_config['layer_type']
        if layer_type == 'dense':
            self.model.add(Dense(
                int(layer_config['num_nodes']),
                activation=layer_config['activation_function']
            ))
        elif layer_type == 'flatten':
            self.model.add(Flatten())
        elif layer_type == 'convolution':
            self.model.add(Conv2D(
                int(layer_config['filters']),
                kernel_size=ast.literal_eval(layer_config['kernel_size']),
                strides=ast.literal_eval(layer_config['stride']),
                activation=layer_config['activation_function']
            ))
        elif layer_type == 'pooling':
            pool_class = MaxPooling2D if layer_config['pooling_type'] == 'max' else AveragePooling2D
            self.model.add(pool_class(
                pool_size=ast.literal_eval(layer_config['pool_size']),
                strides=ast.literal_eval(layer_config['stride'])
            ))
        elif layer_type == 'reshape':
            self.model.add(Reshape(ast.literal_eval(layer_config['target_shape'])))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def fit(self, rdd_data, epochs=1, batch_size=32):
        """Enhanced training method with proper error propagation"""
        try:
            self.spark_model.fit(
                rdd_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X):
        """Prediction with error handling"""
        try:
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def get_parameters(self):
        """Parameter extraction with error handling"""
        try:
            return [layer.get_weights() for layer in self.model.layers]
        except Exception as e:
            logging.error(f"Parameter extraction failed: {str(e)}")
            raise

    def update_parameters(self, new_params):
        """Parameter update with error handling, assuming new_params is a list of arrays"""
        try:
            for layer, weights in zip(self.model.layers, new_params):
                layer.set_weights([np.array(w) for w in weights])
        except Exception as e:
            logging.error(f"Parameter update failed: {str(e)}")
            raise












































































































# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, MaxPooling2D, AveragePooling2D
# import logging
# import numpy as np
# import ast

# # Settingup basic logging configuration
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s:%(message)s')


# def handle_error(error):
#     error_message = f"An error occurred: {error}"
#     logging.error(error_message)
#     # to Add custom error handling logic here


# class CustomCNN:
#     def __init__(self, config):
#         self.config = config
#         self.model = Sequential()

#         try:
#             input_shape = ast.literal_eval(config['input_shape'])
#             self.model.add(Input(shape=input_shape)) 
#             # Adding layers to the model
#             for i in range(len(config['layers'])):
#                 layer = config['layers'][i]
#                 layer_type = layer['layer_type']

#                 if layer_type == 'dense':
#                     self.model.add(Dense(int(layer['num_nodes']), activation=layer['activation_function']))
#                 elif layer_type == 'flatten':
#                     self.model.add(Flatten())
#                 elif layer_type == 'convolution':
#                     self.model.add(Conv2D(
#                         int(layer['filters']),
#                         kernel_size=eval(layer['kernel_size']),
#                         strides=eval(layer['stride']),
#                         activation=layer['activation_function'],
#                     ))
#                 elif layer_type == 'reshape':
#                     self.model.add(Reshape(eval(layer['target_shape'])))
#                 elif layer_type == 'pooling':
#                     if layer["pooling_type"] == "max":
#                         self.model.add(MaxPooling2D(pool_size=eval(layer['pool_size']), strides=eval(layer['stride'])))
#                     elif layer["pooling_type"] == "average":
#                         self.model.add(AveragePooling2D(pool_size=eval(layer['pool_size']), strides=eval(layer['stride'])))

#             output_layer = config['output_layer']
#             self.model.add(Dense(int(output_layer['num_nodes']), activation=output_layer['activation_function']))

#             self.model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=['accuracy'])

#         except Exception as e:
#             handle_error(e)

#     def fit(self, X, y, epochs=1, batch_size=32):
#         try:
#             return self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
#         except Exception as e:
#             handle_error(e)

#     def predict(self, X):
#         try:
#             return self.model.predict(X)
#         except Exception as e:
#             handle_error(e)

#     def get_parameters(self):
#         try:
#             params = {'weights': []}
#             for i, layer in enumerate(self.model.layers):
#                 layer_weights = layer.get_weights()
#                 # Convert each weight array to lists (can't do at once.)
#                 params['weights'].append([w.tolist() for w in layer_weights])
#             return params
#         except Exception as e:
#             handle_error(e)

#     def update_parameters(self, new_params):
#         try:
#             for i, (layer, layer_params) in enumerate(zip(self.model.layers, new_params['weights'])):
#                 # Convert lists back to numpy arrays
#                 layer.set_weights([np.array(w) for w in layer_params])
#         except Exception as e:
#             handle_error(e)
