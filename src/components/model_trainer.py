import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    try:
        def __init__(self):
            self.model_trainer_config=ModelTrainerconfig()

        def InitiateModelTrainer(self, train_array, test_array):
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Train Test is Splited")

            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }
        
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("All Models r2 score is calculated")

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info("found the best model score")
  
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info("found the best model name")

            ## To get the best model
            best_model=models[best_model_name]
            logging.info("found the best model ")

            if best_model_score<0.60:
                raise Exception("Model is not goood")
            logging.info("Best found model on both training and test dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("saved the object")
        
            predicted=best_model.predict(X_test)
            model_r2_score=r2_score(y_test, predicted)

            return best_model, model_r2_score
            logging.info("best model r2 score is calculated")
        
    except Exception as e:
        raise CustomException(e, sys)