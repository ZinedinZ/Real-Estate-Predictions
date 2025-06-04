import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Data Preprocesing
dataset = pd.read_csv("realtor-data.zip.csv")
dataset.pop("prev_sold_date")
dataset.pop("status")
X = dataset.drop("price", axis=1)
y = dataset.iloc[:, 1].values

state_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
state_imputer.fit(X.iloc[:, 4:7])
X.iloc[:, 4:7] = state_imputer.transform(X.iloc[:, 4:7])

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
col = [i for i in range(X.shape[1]) if i not in [4, 5, 6]]
imputer.fit(X.iloc[:, col])
X.iloc[:, col] = imputer.transform(X.iloc[:, col])
print(X.isna().sum())
