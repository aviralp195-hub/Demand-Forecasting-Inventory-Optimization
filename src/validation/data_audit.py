import pandas as pd

df = pd.read_csv("data/raw/master_train.csv")

print("=" * 50)
print("SHAPE")
print("=" * 50)
print(df.shape)

print("\n")

print("=" * 50)
print("INFO")
print("=" * 50)
print(df.info())

print("\n")

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n")

print("=" * 50)
print("DUPLICATES")
print("=" * 50)
print(df.duplicated().sum())

print("\n")

print("=" * 50)
print("COLUMNS")
print("=" * 50)
print(df.columns.tolist())