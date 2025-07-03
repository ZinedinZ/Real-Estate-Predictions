import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataCleaner:

    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None


    def filter_data(self):
        self.dataset = pd.read_csv("realtor-data.csv", nrows=500000)
        self.dataset = self.dataset[self.dataset["price"] < 1000000]

    def drop_columns(self):
        self.dataset.drop(columns=["brokered_by", "status", "street", "prev_sold_date"], inplace=True)

    def add_columns(self):
        self.dataset["price_per_sqft"] = self.dataset["price"] / self.dataset["house_size"]
        pass
    def remove_outliers(self):
        q_low = self.dataset["price_per_sqft"].quantile(0.01)
        q_hi = self.dataset["price_per_sqft"].quantile(0.99)
        self.dataset= self.dataset[(self.dataset["price_per_sqft"] > q_low) & (self.dataset["price_per_sqft"] < q_hi)]
        self.dataset["price_per_sqft"] = self.dataset["price_per_sqft"].mean()

    def split_features(self):
        self.X = self.dataset.drop("price", axis=1)
        self.y = self.dataset["price"]


    def impute_missing(self):
        # Impute Categorical
        state_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        state_imputer.fit(self.X[["state", "city"]])
        self.X[["state", "city"]] = state_imputer.transform(self.X[["state", "city"]])

        # Impute Numerical
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        col = [i for i in range(self.X.shape[1]) if i not in [3, 4]]
        imputer.fit(self.X.iloc[:, col])
        self.X.iloc[:, col] = imputer.transform(self.X.iloc[:, col])


    def fit_encoder(self):
        self.ct = ColumnTransformer(
            transformers=[("encode", OneHotEncoder(handle_unknown='ignore'), ["state", "city"])],
            remainder="passthrough"
        )
        self.X = self.ct.fit_transform(self.X)
        return self.X

    def transform_encoder(self, X):
        return self.ct.transform(X)

    def data_pipeline(self):
        self.filter_data()
        self.drop_columns()
        self.add_columns()
        self.remove_outliers()
        self.split_features()
        self.impute_missing()
