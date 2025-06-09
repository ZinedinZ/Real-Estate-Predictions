import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datacleaner import DataCleaner

class Datatrainer:
    def __init__(self):
        self.dc = DataCleaner()
        self.X = self.dc.encod_data("street", "city", "state")
        self.y = self.dc.y

    def trainer(self):
        X_train, X_test, y_train, y_testn = train_test_split(self.X, self.y, train_size=0.8, random_state=1)
        print(X_train)
        print(X_test)
