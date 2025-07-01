import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datacleaner import DataCleaner
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Datatrainer:
    def __init__(self):
        self.dc = DataCleaner()
        self.dc.data_pipeline()
        self.X = self.dc.X
        self.y = self.dc.y
        self.X = self.dc.encod_data(self.X, "state", "city")

    def trainer(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=42)
        regressor = XGBRegressor()
        regressor.fit(X_train, y_train)
        return regressor

    def predictor(self, dataset):
        regressor = Datatrainer().trainer()
        encoded_dataset = self.dc.encod_data(dataset, "state", "city")
        dataset_pred = regressor.predict(encoded_dataset)
        return f"{dataset_pred[0]:.2f}"
