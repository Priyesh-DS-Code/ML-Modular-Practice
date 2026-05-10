import sys
import os
 
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformed_data(self):
        try:
            logging.info("Data Tansformation is Initiated")
            num_columns=['reading_score', 'writing_score']
            cat_columns=[
                'gender',
                'race_ethnicity', 
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ] 

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoding', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("numrical and categorical pipeline is created")

            preprocessor=ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_columns),
                    ('cat', cat_pipeline, cat_columns)
                ]
            )

            logging.info("preprocessor is created")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def InitiateDataIngestion(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the train and test df is completed")

            preprocessor_obj=self.get_transformed_data()

            output_column=['math_score']

            input_train_df=train_df.drop(columns=output_column, axis=1)
            output_train_df=train_df[output_column]

            input_test_df=test_df.drop(columns=output_column, axis=1)
            output_test_df=test_df[output_column]

            train_df_transformed=preprocessor_obj.fit_transform(input_train_df)
            test_df_transformed=preprocessor_obj.transform(input_test_df)

            logging.info("train and test df is transformed")

            train_arr=np.c_[train_df_transformed, np.array(output_train_df)]
            test_arr=np.c_[test_df_transformed, np.array(output_test_df)]

            save_object(
                file_path=self.data_transformation_config.Preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Saaved the object in pickle file")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.Preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)