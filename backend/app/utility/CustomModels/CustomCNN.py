import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    Reshape,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Dropout,
)
import numpy as np
import ast
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


class SklearnMetricsCallback(Callback):
    """Custom callback to calculate sklearn metrics during training"""
    
    def __init__(self, X_val, y_val, metric_names, is_classification=True):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.metric_names = metric_names
        self.is_classification = is_classification
        self.history = {metric: [] for metric in metric_names}
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions in batches to avoid memory issues
        batch_size = 32
        n_samples = len(self.X_val)
        y_pred_list = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_pred = self.model.predict(self.X_val[i:batch_end], verbose=0)
            y_pred_list.append(batch_pred)
        
        y_pred = np.vstack(y_pred_list)
        
        if self.is_classification:
            # For classification, convert to class labels
            if y_pred.shape[1] > 1:  # Multi-class
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:  # Binary
                y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred_labels = y_pred.flatten()
        
        # Make sure y_val is flattened for comparison
        y_val_flat = self.y_val.flatten() if len(self.y_val.shape) > 1 else self.y_val
        
        # Calculate requested metrics
        for metric in self.metric_names:
            if metric == "accuracy":
                value = accuracy_score(y_val_flat, y_pred_labels)
                self.history[metric].append(value)
                logs[f'val_{metric}'] = value
                
            elif metric == "precision":
                if self.is_classification:
                    # Use weighted average for multi-class
                    avg = 'weighted' if len(np.unique(y_val_flat)) > 2 else 'binary'
                    value = precision_score(y_val_flat, y_pred_labels, average=avg, zero_division=0)
                    self.history[metric].append(value)
                    logs[f'val_{metric}'] = value
                    
            elif metric == "recall":
                if self.is_classification:
                    avg = 'weighted' if len(np.unique(y_val_flat)) > 2 else 'binary'
                    value = recall_score(y_val_flat, y_pred_labels, average=avg, zero_division=0)
                    self.history[metric].append(value)
                    logs[f'val_{metric}'] = value
                    
            elif metric == "f1_score":
                if self.is_classification:
                    avg = 'weighted' if len(np.unique(y_val_flat)) > 2 else 'binary'
                    value = f1_score(y_val_flat, y_pred_labels, average=avg, zero_division=0)
                    self.history[metric].append(value)
                    logs[f'val_{metric}'] = value
                    
            elif metric == "mae":
                if self.is_classification:
                    # For classification, calculate MAE on probabilities
                    value = mean_absolute_error(y_val_flat, y_pred_labels)
                else:
                    value = mean_absolute_error(y_val_flat, y_pred.flatten())
                self.history[metric].append(value)
                logs[f'val_{metric}'] = value
                
            elif metric == "mse":
                if self.is_classification:
                    # For classification, calculate MSE on probabilities
                    value = mean_squared_error(y_val_flat, y_pred_labels)
                else:
                    value = mean_squared_error(y_val_flat, y_pred.flatten())
                self.history[metric].append(value)
                logs[f'val_{metric}'] = value
        
        # Print metrics for this epoch
        metrics_str = " - ".join([f"val_{m}: {self.history[m][-1]:.4f}" for m in self.metric_names])
        print(f"  sklearn metrics: {metrics_str}")


def handle_error(error):
    error_message = f"An error occurred: {error}"
    print(error_message)


class CustomCNN:
    def __init__(self, config):
        self.config = config
        self.model = Sequential()
        self.sklearn_metrics = None
        
        try:
            input_shape = ast.literal_eval(config.get("input_shape", "(128, 128, 1)"))
            self.model.add(Input(shape=input_shape))

            # Adding layers to the model
            for i in range(len(config["layers"])):
                layer = config["layers"][i]
                layer_type = layer["layer_type"]

                if layer_type == "dense":
                    dense_layer = Dense(
                        int(layer["num_nodes"]),
                        activation=layer.get("activation_function", "relu"),
                    )
                    # Add regularization if specified
                    if "regularizer" in layer:
                        reg_config = layer["regularizer"]
                        if reg_config["type"] == "l2":
                            dense_layer.kernel_regularizer = regularizers.l2(
                                float(reg_config["factor"])
                            )
                    self.model.add(dense_layer)

                elif layer_type == "flatten":
                    self.model.add(Flatten())

                elif layer_type == "convolution":
                    conv_layer = Conv2D(
                        int(layer["filters"]),
                        kernel_size=eval(layer.get("kernel_size", "(3, 3)")),
                        strides=eval(layer.get("stride", "(1, 1)")),
                        activation=layer.get("activation_function", "relu"),
                        padding=layer.get("padding", "same"),
                    )
                    self.model.add(conv_layer)

                elif layer_type == "reshape":
                    self.model.add(Reshape(eval(layer["target_shape"])))

                elif layer_type == "pooling":
                    if layer.get("pooling_type", "max") == "max":
                        self.model.add(
                            MaxPooling2D(
                                pool_size=eval(layer.get("pool_size", "(2, 2)")),
                                strides=eval(
                                    layer.get("stride", "None")
                                ),  # None means same as pool_size
                            )
                        )
                    elif layer.get("pooling_type") == "average":
                        self.model.add(
                            AveragePooling2D(
                                pool_size=eval(layer.get("pool_size", "(2, 2)")),
                                strides=eval(layer.get("stride", "None")),
                            )
                        )

                elif layer_type == "batch_norm":
                    self.model.add(BatchNormalization())

                elif layer_type == "dropout":
                    self.model.add(Dropout(float(layer.get("rate", 0.5))))

            # Output layer
            output_layer = config["output_layer"]
            output_dense = Dense(
                int(output_layer["num_nodes"]),
                activation=output_layer.get("activation_function", "linear"),
            )
            if "regularizer" in output_layer:
                reg_config = output_layer["regularizer"]
                if reg_config["type"] == "l2":
                    output_dense.kernel_regularizer = regularizers.l2(
                        float(reg_config["factor"])
                    )
            self.model.add(output_dense)

            # Compile model - only use basic loss, we'll handle metrics separately
            optimizer_config = config.get(
                "optimizer", {"type": "adam", "learning_rate": 0.001}
            )
            if optimizer_config["type"] == "adam":
                optimizer = optimizers.Adam(
                    learning_rate=float(optimizer_config.get("learning_rate", 0.001))
                )
            elif optimizer_config["type"] == "sgd":
                optimizer = optimizers.SGD(
                    learning_rate=float(optimizer_config.get("learning_rate", 0.01))
                )
            else:
                optimizer = optimizer_config["type"]

            # Determine if this is a classification task
            loss_fn = config.get("loss", "mean_squared_error")
            self.is_classification = "categorical" in loss_fn or "sparse_categorical" in loss_fn or "binary" in loss_fn
            
            # Only compile with loss and optimizer, no metrics
            self.model.compile(
                loss=loss_fn,
                optimizer=optimizer
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
            if not hasattr(self.model, "optimizer"):
                print("Error: Model is not compiled")
                return None
            
            # Get requested metrics
            requested_metrics = self.config.get("test_metrics", [])
            
            # Add sklearn metrics callback if we have validation data and metrics
            if validation_data and requested_metrics:
                sklearn_callback = SklearnMetricsCallback(
                    validation_data[0], 
                    validation_data[1], 
                    requested_metrics,
                    self.is_classification
                )
                
                if callbacks is None:
                    callbacks = [sklearn_callback]
                else:
                    callbacks = list(callbacks) + [sklearn_callback]
                
                # Store reference for later
                self.sklearn_metrics = sklearn_callback
            
            # Train the model
            history = self.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Add sklearn metrics to history
            if self.sklearn_metrics:
                for metric in requested_metrics:
                    history.history[f'val_{metric}'] = self.sklearn_metrics.history[metric]
            
            return history
            
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
            params = {"weights": []}
            for i, layer in enumerate(self.model.layers):
                layer_weights = layer.get_weights()
                # Convert each weight array to lists
                params["weights"].append([w.tolist() for w in layer_weights])
            return params
        except Exception as e:
            handle_error(e)

    def update_parameters(self, new_params):
        try:
            for i, (layer, layer_params) in enumerate(
                zip(self.model.layers, new_params["weights"])
            ):
                # Convert lists back to numpy arrays
                layer.set_weights([np.array(w) for w in layer_params])
        except Exception as e:
            handle_error(e)

    # def evaluate(self, x_test, y_test, batch_size=32):
    #     try:
    #         if x_test is None or y_test is None:
    #             print("Error: x_test and y_test cannot be None")
    #             # Return empty dict instead of None to avoid 422 error
    #             return {"loss": 0.0}

    #         # Get loss value (evaluate returns a scalar when no metrics are compiled)
    #         loss_result = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            
    #         # Handle both cases: single value or list
    #         if isinstance(loss_result, (list, tuple)):
    #             loss = float(loss_result[0]) if loss_result else 0.0
    #         else:
    #             loss = float(loss_result)
            
    #         # Initialize results with loss
    #         results = {"loss": loss}
            
    #         # Get predictions for sklearn metrics
    #         y_pred = self.model.predict(x_test, verbose=0)
            
    #         # Ensure y_test is numpy array and properly shaped
    #         if not isinstance(y_test, np.ndarray):
    #             y_test = np.array(y_test)
    #         y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
            
    #         # Convert predictions to labels
    #         if self.is_classification:
    #             if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # Multi-class
    #                 y_pred_labels = np.argmax(y_pred, axis=1)
    #             elif len(y_pred.shape) > 1:  # Binary with 2D output
    #                 y_pred_labels = (y_pred[:, 0] > 0.5).astype(int)
    #             else:  # Binary with 1D output
    #                 y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    #         else:
    #             y_pred_labels = y_pred.flatten()
            
    #         # Calculate only requested metrics
    #         requested_metrics = self.config.get("test_metrics", [])
            
    #         for metric in requested_metrics:
    #             try:
    #                 if metric == "accuracy":
    #                     results[metric] = float(accuracy_score(y_test_flat, y_pred_labels))
                        
    #                 elif metric == "precision":
    #                     if self.is_classification:
    #                         n_classes = len(np.unique(y_test_flat))
    #                         avg = 'weighted' if n_classes > 2 else 'binary'
    #                         results[metric] = float(precision_score(y_test_flat, y_pred_labels, average=avg, zero_division=0))
                            
    #                 elif metric == "recall":
    #                     if self.is_classification:
    #                         n_classes = len(np.unique(y_test_flat))
    #                         avg = 'weighted' if n_classes > 2 else 'binary'
    #                         results[metric] = float(recall_score(y_test_flat, y_pred_labels, average=avg, zero_division=0))
                            
    #                 elif metric == "f1_score":
    #                     if self.is_classification:
    #                         n_classes = len(np.unique(y_test_flat))
    #                         avg = 'weighted' if n_classes > 2 else 'binary'
    #                         results[metric] = float(f1_score(y_test_flat, y_pred_labels, average=avg, zero_division=0))
                            
    #                 elif metric == "mae":
    #                     if self.is_classification:
    #                         results[metric] = float(mean_absolute_error(y_test_flat, y_pred_labels))
    #                     else:
    #                         results[metric] = float(mean_absolute_error(y_test_flat, y_pred.flatten()))
                            
    #                 elif metric == "mse":
    #                     if self.is_classification:
    #                         results[metric] = float(mean_squared_error(y_test_flat, y_pred_labels))
    #                     else:
    #                         results[metric] = float(mean_squared_error(y_test_flat, y_pred.flatten()))
    #             except Exception as metric_error:
    #                 print(f"Warning: Could not calculate {metric}: {metric_error}")
    #                 results[metric] = 0.0
            
    #         # Ensure all values are Python floats, not numpy types
    #         results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
    #                   for k, v in results.items()}
            
    #         return results
            
    #     except Exception as e:
    #         error_message = f"An error occurred in Evaluate Function: {e}"
    #         print(error_message)
    #         # Return a valid dict with default values instead of None
    #         requested_metrics = self.config.get("test_metrics", [])
    #         default_results = {"loss": 0.0}
    #         for metric in requested_metrics:
    #             default_results[metric] = 0.0
    #         return default_results

    def evaluate(self, x_test, y_test, batch_size=32):
        try:
            print(f"[DEBUG] Starting evaluate function")
            print(f"[DEBUG] x_test shape: {x_test.shape if x_test is not None else 'None'}")
            print(f"[DEBUG] y_test shape: {y_test.shape if y_test is not None else 'None'}")
            print(f"[DEBUG] y_test type: {type(y_test)}")
            
            if x_test is None or y_test is None:
                print("Error: x_test and y_test cannot be None")
                # Return empty dict instead of None to avoid 422 error
                return {"loss": 0.0}

            print(f"[DEBUG] About to call model.evaluate...")
            # Get loss value (evaluate returns a scalar when no metrics are compiled)
            loss_result = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            
            print(f"[DEBUG] loss_result type: {type(loss_result)}")
            print(f"[DEBUG] loss_result value: {loss_result}")
            
            # Handle both cases: single value or list
            if isinstance(loss_result, (list, tuple)):
                print(f"[DEBUG] loss_result is list/tuple, length: {len(loss_result)}")
                loss = float(loss_result[0]) if loss_result else 0.0
            else:
                loss = float(loss_result)
            
            print(f"[DEBUG] Processed loss value: {loss}")
            
            # Initialize results with loss
            results = {"loss": loss}
            
            # Get predictions for sklearn metrics
            print(f"[DEBUG] Getting predictions...")
            y_pred = self.model.predict(x_test, verbose=0)
            print(f"[DEBUG] y_pred shape: {y_pred.shape}")
            
            # Ensure y_test is numpy array and properly shaped
            if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
            y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
            print(f"[DEBUG] y_test_flat shape: {y_test_flat.shape}")
            
            # Convert predictions to labels
            print(f"[DEBUG] is_classification: {self.is_classification}")
            if self.is_classification:
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # Multi-class
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    print(f"[DEBUG] Multi-class prediction, y_pred_labels shape: {y_pred_labels.shape}")
                elif len(y_pred.shape) > 1:  # Binary with 2D output
                    y_pred_labels = (y_pred[:, 0] > 0.5).astype(int)
                    print(f"[DEBUG] Binary 2D prediction")
                else:  # Binary with 1D output
                    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
                    print(f"[DEBUG] Binary 1D prediction")
            else:
                y_pred_labels = y_pred.flatten()
                print(f"[DEBUG] Regression prediction")
            
            # Calculate only requested metrics
            requested_metrics = self.config.get("test_metrics", [])
            print(f"[DEBUG] Requested metrics: {requested_metrics}")
            
            for metric in requested_metrics:
                try:
                    print(f"[DEBUG] Calculating {metric}...")
                    if metric == "accuracy":
                        value = accuracy_score(y_test_flat, y_pred_labels)
                        print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                        results[metric] = float(value)
                        
                    elif metric == "precision":
                        if self.is_classification:
                            n_classes = len(np.unique(y_test_flat))
                            avg = 'weighted' if n_classes > 2 else 'binary'
                            value = precision_score(y_test_flat, y_pred_labels, average=avg, zero_division=0)
                            print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                            results[metric] = float(value)
                            
                    elif metric == "recall":
                        if self.is_classification:
                            n_classes = len(np.unique(y_test_flat))
                            avg = 'weighted' if n_classes > 2 else 'binary'
                            value = recall_score(y_test_flat, y_pred_labels, average=avg, zero_division=0)
                            print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                            results[metric] = float(value)
                            
                    elif metric == "f1_score":
                        if self.is_classification:
                            n_classes = len(np.unique(y_test_flat))
                            avg = 'weighted' if n_classes > 2 else 'binary'
                            value = f1_score(y_test_flat, y_pred_labels, average=avg, zero_division=0)
                            print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                            results[metric] = float(value)
                            
                    elif metric == "mae":
                        if self.is_classification:
                            value = mean_absolute_error(y_test_flat, y_pred_labels)
                        else:
                            value = mean_absolute_error(y_test_flat, y_pred.flatten())
                        print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                        results[metric] = float(value)
                            
                    elif metric == "mse":
                        if self.is_classification:
                            value = mean_squared_error(y_test_flat, y_pred_labels)
                        else:
                            value = mean_squared_error(y_test_flat, y_pred.flatten())
                        print(f"[DEBUG] {metric} raw value: {value}, type: {type(value)}")
                        results[metric] = float(value)
                        
                except Exception as metric_error:
                    print(f"[DEBUG] Error calculating {metric}: {metric_error}")
                    import traceback
                    traceback.print_exc()
                    results[metric] = 0.0
            
            # Ensure all values are Python floats, not numpy types
            print(f"[DEBUG] Results before conversion: {results}")
            results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in results.items()}
            print(f"[DEBUG] Final results: {results}")
            print(f"[DEBUG] Results type: {type(results)}")
            
            return results
            
        except Exception as e:
            print(f"[DEBUG] Exception in evaluate: {e}")
            import traceback
            traceback.print_exc()
            error_message = f"An error occurred in Evaluate Function: {e}"
            print(error_message)
            # Return a valid dict with default values instead of None
            requested_metrics = self.config.get("test_metrics", [])
            default_results = {"loss": 0.0}
            for metric in requested_metrics:
                default_results[metric] = 0.0
            print(f"[DEBUG] Returning default results: {default_results}")
            return default_results