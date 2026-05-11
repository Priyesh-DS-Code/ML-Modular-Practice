import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerconfig

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_data_path:sys=os.path.join("artifacts", "train.csv")
    test_data_path:sys=os.path.join("artifacts", "test.csv")
    raw_data_path:sys=os.path.join("artifacts", "data.csv")


class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def InitiateDataIngestion(self):
        logging.info("Data Ingestion Started")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read dataset is completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            logging.info("Directory is made")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("raw data is saved")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("train and test data is saved")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
    

if __name__=='__main__':
    obj=DataIngestion()
    train_path, test_path=obj.InitiateDataIngestion()

    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.InitiateDataIngestion(train_path=train_path, test_path=test_path)

    model_trainer=ModelTrainer()
    best_model_name, model_r2_score=model_trainer.InitiateModelTrainer(train_array=train_array, test_array=test_array)
    print(f"Best Model: {best_model_name}")
    print(f"r2 Score: {model_r2_score}")






