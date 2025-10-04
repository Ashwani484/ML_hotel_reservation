import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
from src.ModelTuner_Classification import ModelTuner
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)
#model training class
class ModelTraining:

    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    #loading and splitting the data
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data splitted sucefully for Model Training")

            return X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data" ,  e)
        
    #training the model using LightGBM    
    
    #saving the trained model    
    def save_model(self,model,model_name):
        try:
            #os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("saving the model")
            
            joblib.dump(model , f"artifacts/models"+f"/{model_name}.pkl")
            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model" ,  e)
    #running the model training pipeline using MLflow for experiment tracking
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our Model Training pipeline")

                logger.info("Starting our MLFLOW experimentation")

                logger.info("Logging the training and testing datset to MLFLOW")
                mlflow.log_artifact(self.train_path , artifact_path="datasets")
                mlflow.log_artifact(self.test_path , artifact_path="datasets")

                X_train,y_train,X_test,y_test =self.load_and_split_data()
                

                best_baseline_model_name,best_model, best_params=ModelTuner.get_best_model(self)

                self.save_model(best_model, best_baseline_model_name)

            

                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and metrics to MLFLOW")
                mlflow.log_params(best_params)
                

                logger.info("Model Training sucesfullly completed")

        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline" ,  e)
        

'''
#final running the model training class        
if __name__=="__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()
        
'''
    

            