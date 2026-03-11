import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset (Study hours vs Pass/Fail)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])

# Model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Prediction
hours = float(input("Enter study hours: "))
prediction = model.predict([[hours]])

if prediction == 1:
    print("Student will Pass")
else:
    print("Student will Fail")