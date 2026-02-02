import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])

pd.DataFrame(X_train_scaled, columns=num_cols).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=num_cols).to_csv("X_test_scaled.csv", index=False)

print("Feature scaling completed")