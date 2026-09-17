import pandas as pd

TRAIN_PATH = "data/raw/master_train.csv"
TEST_PATH = "data/raw/master_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

print(train_df.head())