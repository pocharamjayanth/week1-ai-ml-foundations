import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load your data (make sure the path is correct)
df = pd.read_csv('../data/student_performance.csv')

# 2. Select Features (The inputs)
X = df[['Study_Hours', 'Attendance', 'Sleep_Hours', 'Prev_Scores']]

# 3. Train & Save Regression Model (To predict Grade Point)
y_reg = df['Grade_Point']
reg_model = LinearRegression().fit(X, y_reg)
joblib.dump(reg_model, '../models/regression_model.pkl')

# 4. Train & Save Classification Model (To predict Pass/Fail)
y_clf = df['Pass_Fail']
clf_model = RandomForestClassifier().fit(X, y_clf)
joblib.dump(clf_model, '../models/classification_model.pkl')

print("Both models have been saved in the models folder!")