import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
df = pd.read_csv('data/student_scores.csv')
X = df[['Hours_Studied']]
y = df['Exam_Grade']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model into the 'model' folder
joblib.dump(model, 'model/grade_model.pkl')
print("Success: Model trained and saved in the model folder!")
