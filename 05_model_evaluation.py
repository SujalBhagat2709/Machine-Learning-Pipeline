import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("X_test_scaled.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("model.pkl")

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))