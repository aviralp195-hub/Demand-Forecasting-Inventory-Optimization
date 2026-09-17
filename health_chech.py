import os
import pandas as pd
import joblib

print("="*50)
print("PROJECT HEALTH CHECK")
print("="*50)

# Required files
required_files = [
    "models/best_model.pkl",
    "reports/forecast_output.csv",
    "dashboards/app.py",
    "src/inventory/inventory_optimizer.py",
    "src/tracking/mlflow_tracker.py"
]

print("\nChecking files...")

for file in required_files:
    if os.path.exists(file):
        print(f"[OK] {file}")
    else:
        print(f"[MISSING] {file}")

print("\nChecking model...")

try:
    model = joblib.load("models/best_model.pkl")
    print("[OK] Model loaded")
    print(type(model))
except Exception as e:
    print("[ERROR] Model:", e)

print("\nChecking forecast CSV...")

try:
    df = pd.read_csv(
        "reports/forecast_output.csv",
        engine="python"
    )

    print("[OK] CSV loaded")
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))

except Exception as e:
    print("[ERROR] CSV:", e)

print("\nHealth check complete.")