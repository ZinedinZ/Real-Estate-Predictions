from datacleaner import DataCleaner
from Datatrainer import Datatrainer
import pandas as pd
from flask import Flask, render_template, url_for, request

app = Flask(__name__)
dt = Datatrainer()
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
    user_data = create_dataframe(answers)
    price = dt.predictor(user_data)
    answers += price
    return render_template("home.html", data=answers)

def create_dataframe(data):
        test = pd.DataFrame([{
            "bad": data[0],
            "Bath": data[1],
            "acre_lot": data[2],
            "city": data[3],
            "state": data[4],
            "zip_code": data[5],
            "house_size": data[6]
        }])
        return test

dc = DataCleaner()

if __name__ == "__main__":
    app.run(debug=True)
