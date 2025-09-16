import kagglehub
import torch
from kagglehub import KaggleDatasetAdapter
from src.utils.preprocessing import WeatherPreprocessor

file_path = "weather_classification_data.csv"

def load_data():
    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "nikhil7280/weather-type-classification",
    file_path,
    )

    df.rename(columns={
        "Temperature": "temp",
        "Humidity": "humidity",
        "Wind Speed": "wind_speed",
        "Precipitation (%)": "precipitation",
        "Cloud Cover": "cloud_cover",
        "Atmospheric Pressure": "atm",
        "UV Index": "uv_index",
        "Season": "season",
        "Visibility (km)": "visibility",
        "Location": "location",
    }, inplace=True)


    print(df.head())
    return df

def transform_data(
    df, 
    num_cols, 
    skewed_cols=None, 
    cat_cols=None, 
    target_col="Weather Type", 
    preprocessor=None,
    device="cpu"
):
    """
    Transform raw dataframe into model-ready PyTorch tensors.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        num_cols (list): All numerical columns.
        skewed_cols (list): Columns to apply log transformation.
        cat_cols (list): Categorical columns.
        target_col (str): Name of the target column.
        preprocessor (WeatherPreprocessor, optional): Existing preprocessor for transform.
        device (str): 'cpu' or 'cuda'.

    Returns:
        X_tensor (torch.FloatTensor), y_tensor (torch.LongTensor), preprocessor (WeatherPreprocessor)
    """
    if preprocessor is None:
        # Create and fit a new preprocessor
        preprocessor = WeatherPreprocessor(
            num_cols=num_cols,
            skewed_cols=skewed_cols,
            cat_cols=cat_cols,
            target_col=target_col
        )
        X_processed, y_encoded = preprocessor.fit_transform(df)
    else:
        # Use existing preprocessor
        X_processed, y_encoded = preprocessor.transform(df)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)
    
    return X_tensor, y_tensor, preprocessor

