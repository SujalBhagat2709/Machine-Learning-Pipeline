import pandas as pd
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("X_train_scaled.csv")
y_train = pd.read_csv("y_train.csv")

model = LinearRegression()
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "model.pkl")

print("Model training completed")