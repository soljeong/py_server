from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

class ModelHandler:
    def __init__(self):
        self.models = {}

    def set_models(self, models_dict):
        self.models = models_dict

    def train_model(self, target_col, model):
        self.models[target_col] = model

    def predict(self, target_col, input_data):
        if target_col not in self.models:
            raise ValueError(f"Model for target column '{target_col}' is not trained. Train the model first.")
        return self.models[target_col].predict(input_data)
    
    def get_model(self, target_col):
        if target_col not in self.models:
            raise ValueError(f"Model for target column '{target_col}' is not trained. Train the model first.")
        return self.models[target_col]