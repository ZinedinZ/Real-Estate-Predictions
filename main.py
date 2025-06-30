from datacleaner import DataCleaner
from Datatrainer import Datatrainer
import pandas as pd
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    answers = []
    data = ["state", "city", "zip_code", "house_size", "land_size", "bad", "bath"]
    for i in data:
        i = request.form[i]
        if i:
            answers.append(i)
        print(answers)
    return render_template("home.html", data=answers)

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
