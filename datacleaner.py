import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataCleaner:

    def __init__(self):
        self.dataset = pd.read_csv("realtor-data.zip.csv", nrows=10000)
        self.dataset.pop("prev_sold_date")
        self.dataset.pop("status")
        self.X = self.dataset.drop("price", axis=1)
        self.y = self.dataset.iloc[:, 1].values

    def clean_data(self):
        # Data Preprocesing
        state_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        state_imputer.fit(self.X.iloc[:, 4:7])
        self.X.iloc[:, 4:7] = state_imputer.transform(self.X.iloc[:, 4:7])

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        col = [i for i in range(self.X.shape[1]) if i not in [4, 5, 6]]
        imputer.fit(self.X.iloc[:, col])
        self.X.iloc[:, col] = imputer.transform(self.X.iloc[:, col])
        print("Ima li još NaN vrijednosti:", self.X.isna().sum().sum())

    def encod_data(self,*columns):
        print("Kolone sa NaN vrijednostima:")
        print(self.X.columns[self.X.isna().any()])
        ct = ColumnTransformer(transformers=[("encode", OneHotEncoder(), list(columns))], remainder="passthrough")
        self.X = ct.fit_transform(self.X)
        return self.X