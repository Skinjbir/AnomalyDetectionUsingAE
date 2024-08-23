import logging
import pandas as pd

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logging.info(f"Successfully ingested data from {self.data_path}")
            return df
        except Exception as e:
            logging.error(f"Error while ingesting data from {self.data_path}: {e}")
            raise e

def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        ingestor = IngestData(data_path)
        df = ingestor.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e

