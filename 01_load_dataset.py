import pandas as pd

df = pd.read_csv("step3_transformed.csv")

print("Dataset loaded")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())