import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class DataCleaner:
    def __init__(self, data: pd.DataFrame, scale_method: str = 'standardize') -> None:
        if scale_method not in ['standardize', 'normalize']:
            raise ValueError("Invalid scale_method. Choose 'standardize' or 'normalize'.")
        self.scale_method = scale_method
        self.label_encoders = {}
        self.data = data

    def encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            self.label_encoders[col] = label_encoder
            logging.info(f"Encoded column '{col}' with labels: {label_encoder.classes_}")
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            initial_shape = df.shape
            df = df.drop_duplicates()
            logging.info(f"Removed duplicates: {initial_shape} -> {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error during dropping duplicates: {e}")
            raise e

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.isnull().sum().sum() > 0:
                df = df.fillna(df.mean(numeric_only=True))
                logging.info("Filled missing values with column means.")
            return df
        except Exception as e:
            logging.error(f"Error during handling missing values: {e}")
            raise e

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.scale_method == 'standardize':
                scaler = StandardScaler()
            elif self.scale_method == 'normalize':
                scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            logging.info(f"Scaled features using {self.scale_method}.")
            return pd.DataFrame(X_scaled, columns=X.columns)
        except Exception as e:
            logging.error(f"Error during scaling features: {e}")
            raise e

    def clean_data(self) -> pd.DataFrame:
        try:
            logging.info("Starting data cleaning and encoding.")
            df = self.drop_duplicates(self.data)
            df = self.handle_missing_values(df)
            df = self.encode_columns(df)

            X = df.drop(columns=['Class'])
            y = df['Class']

            X_scaled = self.scale_features(X)
            df_cleaned = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

            logging.info("Data cleaning completed successfully.")
            return df_cleaned
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise e
