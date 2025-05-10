from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open("bmi_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    bmi = None
    if request.method == "POST":
        data = {
            "Gender": request.form["gender"],
            "Height": float(request.form["height"]),
            "Weight": float(request.form["weight"]),
            "Age": int(request.form["age"]),
            "Activity_Level": request.form["activity"],
            "Smoker": request.form["smoker"],
            "Region": request.form["region"]
        }

        df = pd.DataFrame([data])
        x_enc = preprocessor.transform(df)
        bmi = model.predict(x_enc)[0]

    return render_template("index.html", bmi=bmi)

if __name__ == "__main__":
    app.run(debug=True)
