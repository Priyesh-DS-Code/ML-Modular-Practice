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
from src.utils import save_object, evaluate_modelss

@dataclass
class ModelHyperparameterTunningConfig:
    hyperpara_trained_model_file_path=os.path.join('artifacts', 'hyperparameter_model.pkl')

class ModelHyperparameterTunning:
    try:
        def __init__(self):
            self.model_hyperparametertunning_config=ModelHyperparameterTunningConfig()

        def InitiateHyperparaModelTrainer(self, train_array, test_array):
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
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
        
            model_report:dict=evaluate_modelss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
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
                file_path=self.model_hyperparametertunning_config.hyperpara_trained_model_file_path,
                obj=best_model
            )
            logging.info("saved the object")
        
            predicted=best_model.predict(X_test)
            model_r2_score=r2_score(y_test, predicted)

            return best_model, model_r2_score
            logging.info("best model r2 score is calculated")
        
    except Exception as e:
        raise CustomException(e, sys)