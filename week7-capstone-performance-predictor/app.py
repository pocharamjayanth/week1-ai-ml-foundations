from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models from the 'models' folder
reg_model = joblib.load('models/regression_model.pkl')
clf_model = joblib.load('models/classification_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs (Study Hours, Attendance, Sleep Hours, Prev Scores)
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Get predictions
    grade = reg_model.predict(final_features)[0]
    status_code = clf_model.predict(final_features)[0]
    status = "PASS" if status_code == 1 else "FAIL"
    
    return render_template('index.html', grade=round(grade, 2), status=status)

if __name__ == "__main__":
    app.run(debug=True)