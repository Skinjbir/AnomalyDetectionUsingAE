import logging
from stages.ingest_data import IngestData
from stages.clean_data import DataCleaner
from stages.split_data import DataSplitter
from stages.feature_engineering import FeatureEngineer
from stages.train_model import ModelTrainer
import joblib
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Ingest Data
    data_path = 'Train_data.csv'  # Adjust the path to your data file
    ingestor = IngestData(data_path)
    df = ingestor.get_data()
    
    # Clean Data
    cleaner = DataCleaner(scale_method='standardize')  # Or 'normalize'
    X= cleaner.clean_data(df)
    
    # Feature Selection &  Deliver Data
    feature_engineer = FeatureEngineer(X)
    selected_features, y_target = feature_engineer.select_features(X, 'class', n_features_to_select=10)
    X_selected, target = feature_engineer.deliver_data(X, selected