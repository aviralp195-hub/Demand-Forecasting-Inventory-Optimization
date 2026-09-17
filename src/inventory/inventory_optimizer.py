# src/inventory/inventory_optimizer.py

import pandas as pd

forecast = pd.read_csv(
    "reports/forecast_output.csv"
)

forecast["Safety_Stock"] = (
    forecast["Predicted_Sales"] * 0.2
)

forecast["Reorder_Point"] = (
    forecast["Forecast"]
    + forecast["Safety_Stock"]
)

forecast.to_csv(
    "reports/inventory_plan.csv",
    index=False
)

print("Inventory plan created")