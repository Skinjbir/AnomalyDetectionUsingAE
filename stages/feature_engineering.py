import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from typing import Tuple, List

class FeatureEngineer:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.label_encoders = {}
        self.encoded_columns = []
    
    def get_data(self) -> pd.DataFrame:
        logging.info("Feature Engineering the data")
        try:
            logging.info("Successfully got the data to work with")
            return self.dataset
        except Exception as e:
            logging.error(f"Error while getting the data: {e}")
            raise e

    def encode_labels(self) -> pd.DataFrame:
        try:
            for column in self.dataset.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                self.dataset[column] = le.fit_transform(self.dataset[column])
                self.label_encoders[column] = le
                self.encoded_columns.append(column)
            logging.info(f"Encoded columns: {self.encoded_columns}")
            return self.dataset
        except Exception as e:
            logging.error(f"Error while encoding labels: {e}")
            raise e

    def get_encoded_columns(self) -> list:
        return self.encoded_columns

    def select_features(self, X: pd.DataFrame, target: str, n_features_to_select: int = 30, bypass_selection: bool = False) -> Tuple[List[str], pd.Series]:
        X_train = X.drop([target], axis=1)
        y_train = X[target]
        try:
            if bypass_selection:
                logging.info("Bypassing feature selection, returning target")
                return X_train.columns.tolist(), y_train  # Return all features if bypassed
            
            logging.info("Selecting important features using RandomForestClassifier and RFE")
            rfc = RandomForestClassifier()
            rfe = RFE(estimator=rfc, n_features_to_select=n_features_to_select)
            rfe.fit(X_train, y_train)
            
            feature_map = [(i, v) for i, v in zip(rfe.get_support(), X_train.columns)]
            selected_features = [v for i, v in feature_map if i]
        
            logging.info(f"Selected features: {selected_features}")
            return selected_features, y_train
        
        except Exception as e:
            logging.error(f"Error while selecting features: {e}")
            raise e

    def deliver_data(self, dataset: pd.DataFrame, features: list, target) -> pd.DataFrame:
        try:
            logging.info("Delivering data with selected features")
            return dataset, target,
        except KeyError as e:
            logging.error(f"One or more features are not in the dataset: {e}")
            raise e
        except Exception as e:
            logging.error(f"Error while delivering data: {e}")
            raise e
