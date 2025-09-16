# src/utils/preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def log1p_transform(X):
    return np.log1p(X)

class WeatherPreprocessor:
    """
    Preprocessing pipeline for weather dataset using FunctionTransformer:
    - Log-transform specified skewed numerical features
    - Scale all numerical features
    - One-hot encode categorical features
    - Label encode target
    """
    def __init__(self, num_cols, skewed_cols=None, cat_cols=None, target_col="Weather Type"):
        self.num_cols = num_cols
        self.skewed_cols = skewed_cols if skewed_cols is not None else []
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.target_col = target_col

        # Log-transform pipeline only for skewed columns
        self.log_pipeline = Pipeline([
            ('log', FunctionTransformer(
                log1p_transform,
                validate=False
            )),
            ('scaler', StandardScaler())
        ])

        # Standard scaling pipeline for remaining numerical columns
        self.num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        # Combine using ColumnTransformer
        transformers = []
        if self.skewed_cols:
            transformers.append(('log_num', self.log_pipeline, self.skewed_cols))
        remaining_cols = [c for c in self.num_cols if c not in self.skewed_cols]
        if remaining_cols:
            transformers.append(('num', self.num_pipeline, remaining_cols))
        if self.cat_cols:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), self.cat_cols))

        self.preprocessor = ColumnTransformer(transformers)

        # Target label encoder
        self.label_encoder = LabelEncoder()

    def fit_transform(self, df):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_processed = self.preprocessor.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        return X_processed, y_encoded

    def transform(self, df):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_processed = self.preprocessor.transform(X)
        y_encoded = self.label_encoder.transform(y)
        return X_processed, y_encoded
    
    def transform_predict(self, df):
        """
        Transform incoming DataFrame for prediction (without target column)
        Returns processed features only.
        """
        X_processed = self.preprocessor.transform(df)
        return X_processed

    def inverse_transform_target(self, y_pred):            
        return self.label_encoder.inverse_transform(y_pred)