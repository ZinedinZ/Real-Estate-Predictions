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
        self.dc.fit_encoder()
        self.X = self.dc.X
        self.y = self.dc.y
        self.regressor = self.trainer()

    def trainer(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=42)
        regressor = XGBRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R^2: {r2:.2f}")
        return regressor

    def predictor(self, dataset):
        encoded_dataset = self.dc.transform_encoder(dataset)
        print("ENCODED INPUT:\n", encoded_dataset)

        dataset_pred = self.regressor.predict(encoded_dataset)
        print("PREDICTION:", dataset_pred)


        return f"{dataset_pred[0]:.2f}"
