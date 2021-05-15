"""
Project: Ung Dung Machine Learning Chuan Doan Kha Nang Mac Benh Tim trong 10 nam toi
Class : Python
Teacher : Ts.Vu Tien Dung
Group 5, Ms Data Science K3
"""

# import library
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import glob
import os
import pandas as pd
from datasets import Datasets


def get_transform(nparray):
    data_file = "framingham.csv"
    numeric_var = [
        "age",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    ]
    level_var = ["education"]
    category_var = [
        "male",
        "currentSmoker",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
    ]
    target = ["TenYearCHD"]

    # Create Data object
    data = Datasets(
        data_file=data_file,
        cat_cols=category_var,
        num_cols=numeric_var,
        level_cols=level_var,
        label_col=target,
        train=True,
    )
    return data.preprocess_newdata(nparray)


def get_predict(data):
    data = get_transform(np.array(data))
    list_file = glob.glob(os.path.join("./log/best_model", "*"))
    list_model = []
    for l in list_file:
        if l.split("/")[-1].split(".")[-1] == "pkl":
            list_model.append(l)
    model_path = list_model[0]
    print(f"Loading..model {model_path}")
    clf = joblib.load(model_path)
    pred = clf.predict_proba(data)[:, 1][0]
    return pred


app = Flask(__name__)  # Initialize the flask App


@app.route("/")  # Homepage
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # get value from user input
    init_features = [float(x) for x in request.form.values()]
    X_new = np.array([init_features])

    # predict
    y_pred = get_predict(X_new)
    if y_pred > 0.5:
        return render_template(
            "index2.html",
            prediction_text="Your Risk score is {} . High risk, please care more about your health".format(
                str(round(y_pred, 2))
            ),
        )  # rendering the predicted result
    else:
        return render_template(
            "index2.html",
            prediction_text="Your Risk score is {} . Congras, let's keep your healthy hobbies!".format(
                str(round(y_pred, 2))
            ),
        )  # rendering the predicted result


if __name__ == "__main__":
    app.debug = False
    from werkzeug.serving import run_simple

    run_simple("localhost", 5000, app)
