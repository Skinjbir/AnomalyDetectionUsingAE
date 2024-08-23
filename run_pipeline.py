import logging
import pandas as pd
import os
import mlflow
from stages.ingest_data import IngestData
from stages.clean_data import DataCleaner
from stages.split_data import DataSplitter
from stages.train_model import ModelTrainer
from stages.create_model import create_autoencoder
from stages.evaluate_model import detect_anomalies, evaluate_anomalies
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Start MLflow run
        with mlflow.start_run() as run:
            logging.info("MLflow run started.")

            # Ingest Data
            data_path = 'creditcard.csv'
            df = pd.read_csv(data_path)
            logging.info("Data ingested successfully.")
            logging.debug(f"Data sample: {df.head()}")

            # Clean Data
            dataCleaner = DataCleaner(scale_method='normalize', data=df)
            X = dataCleaner.clean_data()
            logging.info("Data cleaned successfully.")

            # Split Data
            dataSplitter = DataSplitter(X)
            X_train, X_test, y_train, y_test = dataSplitter.split_data_v2()
            logging.info("Data split successfully.")
            logging.debug(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

            # Create Autoencoder Model
            input_dim = X_train.shape[1]
            autoencoder = create_autoencoder(input_dim=input_dim)

            
            mlflow.log_param("model_type", "autoencoder")
            mlflow.log_param("input_dim", input_dim)

            # Train Autoencoder Model
            trainer = ModelTrainer(training_set=X_train, validation_set=X_test, autoencoder=autoencoder)
            trainer.train_model(batch_size=128, epochs=100)  

            trained_autoencoder = trainer.get_trained_model()
            logging.info("Autoencoder model trained successfully.")

            # Ensure the directory exists
            os.makedirs("./saved_model", exist_ok=True)

            # Save the trained Autoencoder Model
            model_path = "./saved_model/model.plk"
            trainer.save_trained_model(model_path)
            logging.info("Autoencoder model saved successfully.")

            # Log the trained model as an artifact TOADD

            # Detect Anomalies
            outliers, errors, threshold = detect_anomalies(trained_autoencoder, X_test, percentile=99.45, metric='mse')

            # Evaluate Model Performance
            metrics = evaluate_anomalies(y_test, outliers, errors)
            logging.info(f"Evaluation Metrics: {metrics}")

            # Log metrics
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("auc", metrics['auc'])

            logging.info("MLflow run completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
