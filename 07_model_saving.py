import joblib

model = joblib.load("model.pkl")

joblib.dump(model, "final_model.pkl")

print("Final trained model saved as final_model.pkl")
