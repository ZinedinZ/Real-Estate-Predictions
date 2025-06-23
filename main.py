from datacleaner import DataCleaner
from Datatrainer import Datatrainer
import pandas as pd

test = pd.DataFrame([{
            "bad": 3.0,
            "Bath":2.0,
            "acre_lot":0.12,
            "city": "Adjuntas",
            "state": "Puerto Rico",
            "zip_code": 601.0,
            "hose_size": 920.0

}])
dc = DataCleaner()
dt = Datatrainer()
dt.predictor(test)
