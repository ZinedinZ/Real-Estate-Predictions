from datacleaner import DataCleaner
from Datatrainer import Datatrainer
import pandas as pd
from flask import Flask, render_template, url_for, request

app = Flask(__name__)
dt = Datatrainer()
dc = DataCleaner()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    answers = []
    data = ["state", "city", "zip_code", "house_size", "land_size", "bed", "bath"]
    for i in data:
        i = request.form[i]
        if i:
            answers.append(i)
    user_data = create_dataframe(answers)
    price = dt.predictor(user_data)
    answers.append(price)
    return render_template("home.html", data=answers)

def create_dataframe(data):
        test = pd.DataFrame([{
            "bed": data[5],
            "bath": data[6],
            "acre_lot": float(data[4])/4046.85642,
            "city": data[1],
            "state": data[0],
            "zip_code": data[2],
            "house_size": float(data[3])/0.092903
        }])
        return test

if __name__ == "__main__":
    app.run(debug=True)
