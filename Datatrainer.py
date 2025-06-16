import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datacleaner import DataCleaner
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Datatrainer:
    def __init__(self):
        self.dc = DataCleaner()
        self.dc.clean_data()
        self.X = self.dc.encod_data("street", "city", "state")
        self.y = self.dc.y

    def trainer(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=1)
        regressor = XGBRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")


