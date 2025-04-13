from .CustomModels.LandMarkSVM import LandMarkSVM
from .CustomModels.CustomSVM import CustomSVM
from .CustomModels.LinearRegression import LinearRegression
from .CustomModels.MultiLayerPerceptron import MultiLayerPerceptron
from .CustomModels.CustomCNN import CustomCNN

import json

model_classes = {
    "LinearRegression": LinearRegression,
    "SVM": CustomSVM,
    "LandMarkSVM": LandMarkSVM,
    "multiLayerPerceptron": MultiLayerPerceptron,
    "CNN": CustomCNN
}

def model_instance_from_config(modelConfig):
    try:
        model_name = modelConfig["model_name"]
        config = modelConfig["model_info"]
        model_class = model_classes.get(model_name)

        if model_class is None:
            raise ValueError(f"Unknown model: {model_name}")

        model_instance = model_class(config)
        return model_instance

    except Exception as e:
        print(f"Error creating model instance: {e}")
        return None
