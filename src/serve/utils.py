import mlflow.pytorch
import torch
import pandas as pd
from src.data.etl import transform_data, load_data

# Load default MLflow model
def load_model(run_id):
    return mlflow.pytorch.load_model(f"runs:/{run_id}/weather_mlp_model")

# Transform incoming user data
def preprocess_input(df, preprocessor, device="cpu"):
    X_tensor, _, _ = transform_data(
        df,
        num_cols=preprocessor.num_cols,
        skewed_cols=preprocessor.skewed_cols,
        cat_cols=preprocessor.cat_cols,
        target_col=preprocessor.target_col,
        preprocessor=preprocessor,
        device=device
    )
    return X_tensor