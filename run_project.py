import subprocess

commands = [

    "python health_check.py",

    "python src/inventory/inventory_optimizer.py",

    "python src/tracking/mlflow_tracker.py"

]

for cmd in commands:

    print("\nRUNNING:", cmd)

    result = subprocess.run(
        cmd,
        shell=True
    )

    if result.returncode != 0:
        print("FAILED:", cmd)
        break

print("\nPROJECT EXECUTION FINISHED")