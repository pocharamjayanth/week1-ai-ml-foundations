from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the brain you trained
model = joblib.load('model/grade_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get hours from the form
    hours = float(request.form['hours'])
    # Predict the score
    prediction = model.predict(np.array([[hours]]))[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
