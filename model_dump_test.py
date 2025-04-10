from backend.app.utility.CustomModels import CustomCNN



def get_model_config(session_id):
    """Generate sample model configuration matching data structure"""
    return {
        "organisation_name": "test-clinic",
        "dataset_info": {
            "client_filename": "test_data.parquet",
            "output_columns": ["pct_2013"],
            "task_id": 3,
            "metric": "MAE"
        },
        "model_name": "CNN",
        "model_info": {
            "input_shape": "(128,128,1)",
            "output_layer": {
                "num_nodes": "1",
                "activation_function": "sigmoid"
            },
            "loss": "mse",
            "optimizer": "adam",
            "test_metrics": ["mae"],
            "layers": [
                {
                    "layer_type": "convolution",
                    "filters": "8",
                    "kernel_size": "(3,3)",
                    "stride": "(1,1)",
                    "activation_function": "relu"
                },
                {
                    "layer_type": "pooling",
                    "pooling_type": "max",
                    "pool_size": "(2,2)",
                    "stride": "(2,2)"
                },
                {
                    "layer_type": "flatten"
                },
                {
                    "layer_type": "dense",
                    "num_nodes": "64",
                    "activation_function": "relu"
                }
            ]
        }
    } 

import pickle
def test_model_dump():
    model = CustomCNN(
            config=get_model_config(1),
            # spark_context=spark.sparkContext,  # Pass actual Spark context
            mode='asynchronous',
            num_workers=2
        )
    
    try:
        serialized_model = pickle.dumps(model)
        print("Model is serializable!")
    except Exception as e:
        print("Model serialization failed:", e)

    # Optionally, you can dump the pickle to a file:
    with open("compiled_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Model dumped to 'compiled_model.pkl' successfully.")


test_model_dump()