"""
Loan Risk Prediction
Author: Mohd Kaif

This project demonstrates a basic machine learning workflow
to assess loan default risk. The focus is on understanding
risk factors, feature selection, and classification logic.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample dataset (illustrative)
data = {
    "income": [30000, 50000, 40000, 80000, 60000, 70000],
    "loan_amount": [200000, 300000, 250000, 400000, 350000, 380000],
    "credit_score": [550, 620, 580, 720, 680, 700],
    "default_risk": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[["income", "loan_amount", "credit_score"]]
y = df["default_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Loan Risk Prediction Model Results")
print(classification_report(y_test, predictions))
