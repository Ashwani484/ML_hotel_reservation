from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining

from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from config.buildconfig import generate_config
import pandas as pd

def run():
    try:
        logger = get_logger(__name__)
        logger.info("Training Pipeline Started")

        ##to build the config file from input dataset whcih contains categorical and numerical columns
        generate_config(pd.read_csv(RAW_FILE_PATH))

        logger.info("Config file generated successfully")

        ### 1. Data Ingestion

        data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
        data_ingestion.run()

        ### 2. Data Processing

        processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
        processor.process() 

        ### 3. Model Training

        trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
        trainer.run()

        logger.info("Training Pipeline Completed Successfully")

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in training pipeline: {e}")
        raise CustomException("Error in training pipeline", e)  
    
if __name__=="__main__":
    run()   