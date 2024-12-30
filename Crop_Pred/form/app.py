import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


flask_app = Flask(__name__)
model = pickle.load(open("C://Users//ks520//Desktop//New folder//model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("prediction.html", prediction = prediction)

if __name__ == "__main__":
    flask_app.run(debug=True)