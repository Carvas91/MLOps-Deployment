import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocesor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_columns = ['writing_sscore', 'reading_score']
            categorical_columns = ['gender', 
                                   'race_ethnicity', 
                                   'parental_level_of_education', 
                                   'lunch', 
                                   'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("standard scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("One Hot Encoder", OneHotEncoder()),
                    ("standard scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical',num_pipeline, numeric_columns ),
                    ('caterical',cat_pipeline, categorical_columns )
                ]
            )
            

        except:
            pass