from datacleaner import DataCleaner
from Datatrainer import Datatrainer
from userinterface import Userinterface
import pandas as pd
from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

test = pd.DataFrame([{
            "bad": 3.0,
            "Bath":2.0,
            "acre_lot":0.12,
            "city": "Adjuntas",
            "state": "Puerto Rico",
            "zip_code": 601.0,
            "house_size": 920.0

}])
dc = DataCleaner()
dt = Datatrainer()
dt.predictor(test)

if __name__ == "__main__":
    app.run(debug=True)
