# app.py

from flask import Flask, render_template, request
import pickle
from preprocessing import preprocess_input

app = Flask(__name__)

# Load model
model = pickle.load(open("model/xgb_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.to_dict()
    final_input = preprocess_input(user_input)
    prediction = model.predict(final_input)[0]
    return render_template("index.html", prediction_text=f"Predicted Flight Price: ₹{round(prediction, 2)}")

if __name__ == "__main__":
    app.run(debug=True)
