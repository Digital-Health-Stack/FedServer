from .CustomModels.LandMarkSVM import LandMarkSVM
from .CustomModels.CustomSVM import CustomSVM
from .CustomModels.LinearRegression import LinearRegression
from .CustomModels.MultiLayerPerceptron import MultiLayerPerceptron
from .CustomModels.CustomCNN import CustomCNN
from .CustomModels.LogisticRegression import LogisticRegression
from .CustomModels.DecisionTree import DecisionTree
from .CustomModels.RandomForest import RandomForest
from .CustomModels.CustomSVR import CustomSVR
from .CustomModels.XGBoostRegressor import XGBoostRegressor
from .CustomModels.LassoRegression import LassoRegression
from .CustomModels.RidgeRegression import RidgeRegression
from .CustomModels.KNNClassifier import KNNClassifier
from .CustomModels.knn_regressor import KNNRegressor
from .CustomModels.adaboost import AdaBoost
from .CustomModels.lightgbm_regressor import LightGBMRegressor
from .CustomModels.lightgbm_classifier import LightGBMClassifier
from .CustomModels.naive_bayes import NaiveBayes
from .CustomModels.random_forest_regressor import RandomForestRegressor
from .CustomModels.xgboost_classifier import XGBoostClassifier


import json

model_classes = {
    "LinearRegression": LinearRegression,
    "SVM": CustomSVM,
    "SVR": CustomSVR,
    "LandMarkSVM": LandMarkSVM,
    "multiLayerPerceptron": MultiLayerPerceptron,
    "CNN": CustomCNN,
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTree,
    "RandomForest": RandomForest,
    "XGBoostRegressor": XGBoostRegressor,
    "LassoRegression": LassoRegression,
    "RidgeRegression": RidgeRegression,
    "KNNClassifier": KNNClassifier,
    "KNNRegressor": KNNRegressor,
    "AdaBoost": AdaBoost,
    "LightGBMRegressor": LightGBMRegressor,
    "LightGBMClassifier": LightGBMClassifier,
    "NaiveBayes": NaiveBayes,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBoostClassifier": XGBoostClassifier,
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
