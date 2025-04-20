from fastapi import APIRouter, HTTPException
import os
from utility.auth import role
from fastapi import Depends
from models.User import User
import subprocess
import sys
from utility.auth import verify_token
from utility.db import get_db
from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession

temporary_router = APIRouter()

@temporary_router.get('/check')
def check():
    return {"message": "Everyone can access it!"}

@temporary_router.get('/check-client')
def check_client(client: User = Depends(role("client"))):
    return {"message": "Only clients can access it!"}

@temporary_router.get('/check-admin')
def check_admin(admin: User = Depends(role("admin"))):
    return {"message": "Only admins can access it!"}


@temporary_router.get("/check-current_user")
def check_user(token: str):
    current_user = verify_token(token)
    return current_user


@temporary_router.get('/check-session-creation')
def check_session_creation(db: Session = Depends(get_db)):
    # Dummy federated_info data
    federated_info = {
        "organisation_name": "Prashant 17 Apr, 10:40 AM",
        "model_name": "CNN",
        "model_info": {
            "input_shape": "(128,128,1)",
            "output_layer": {
                "num_nodes": "1",
                "activation_function": "sigmoid"
            },
            "loss": "mae",
            "optimizer": "adam",
            "test_metrics": [
                "mse",
                "mae"
            ],
            "layers": [
                {
                    "layer_type": "convolution",
                    "filters": "32",
                    "kernel_size": "(3,3)",
                    "stride": "(2,2)",
                    "activation_function": "relu"
                },
                {
                    "layer_type": "pooling",
                    "pooling_type": "average",
                    "pool_size": "(2,2)",
                    "stride": "(2,2)"
                },
                {
                    "layer_type": "convolution",
                    "filters": "16",
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
                }
            ]
        },
        "dataset_info": {
            "client_filename": "health_client.parquet",
            "server_filename": "health_server.parquet",
            "task_id": "5",
            "task_name": "Bone Age Prediction",
            "metric": "MAE",
            "output_columns": [
                "pct_2013"
            ]
        },
        "expected_results": {
            "std_mean": "9",
            "std_deviation": "0.03"
        }
    }
    
    # Simulating a valid admin_id, assuming an admin user with id 1 exists
    admin_id = 1

    # Insert the federated session into the database
    try:
        federated_session = FederatedSession(
            federated_info=federated_info,
            admin_id=admin_id
        )
        db.add(federated_session)
        db.commit()
        db.refresh(federated_session)
        return {
            "message": "Federated session created successfully",
            "session_id": federated_session.id
        }
    except Exception as e:
        db.rollback()
        print("Error creating federated session:", str(e))  # or use logging
        raise HTTPException(status_code=500, detail="Failed to create session")
    


@temporary_router.get('/get-session/{session_id}')
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session.id,
        "federated_info": session.federated_info,
        "admin_id": session.admin_id,
    }




import numpy as np
from statsmodels.stats.power import TTestIndPower

def cohens_d(mean1, mean2, std1, std2):
    """
    Calculate Cohen's d for effect size between two independent samples.
    """
    # Pooled standard deviation
    s = np.sqrt((std1**2 + std2**2) / 2)
    if s == 0:
        return 0
    d = (mean2 - mean1) / s
    return 1/d

def estimate_sample_size(effect_size, alpha=0.05, power=0.80):
    """
    Estimate required sample size per group for two-sample t-test using Cohen's d.
    """
    analysis = TTestIndPower()
    if effect_size == 0:
        return None  # Cannot compute sample size with zero effect size
    sample_size = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(sample_size))

def get_num_predictors_from_config(model_config):
    """
    Extract number of filters from the last convolution layer in the CNN model config.
    If no convolution layer is present, it will return 0.
    """
    # Loop through the layers in reverse order to find the last convolutional layer
    for layer in reversed(model_config['model_info']['layers']):
        if layer['layer_type'] == 'convolution':
            return int(layer['filters'])  # Return the number of filters (predictors)
    
    return 0  # If no convolution layer is found, return 0 predictors

def calculate_required_data_points(num_predictors, baseline_mean, baseline_std, new_mean, new_std, alpha=0.05, power=0.80):
    """
    Calculate the required number of data points as payment in a federated learning system,
    based on the number of predictors and effect size.
    """
    # Get the number of predictors from the model configuration
    

    # Calculate effect size (Cohen's d)
    effect_size = cohens_d(baseline_mean, new_mean, baseline_std, new_std)

    # Adjust alpha for multiple tests (Bonferroni correction)
    alpha_adjusted = alpha / num_predictors if num_predictors > 0 else alpha

    # Estimate the required sample size
    required_sample_size = estimate_sample_size(effect_size, alpha=alpha_adjusted, power=power)
    return required_sample_size

@temporary_router.get("/check-price")
def check_price(predictors:float, baseline_mean: float, baseline_std: float, new_mean:float, new_std: float):
    return calculate_required_data_points(predictors, baseline_mean, baseline_std, new_mean, new_std)