# generate_forecast.py

import pandas as pd
import joblib

model = joblib.load(
    "models/best_model.pkl"
)

df = pd.read_csv(
    "data/processed/final_features.csv"
)

X = df.drop(
    columns=["Sales"]
)

preds = model.predict(X)

output = df.copy()

output["Forecast"] = preds

output.to_csv(
    "reports/forecast_output.csv",
    index=False
)

print("Forecast file created")