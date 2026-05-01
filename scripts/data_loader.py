import pandas as pd

def load_data(path="../data"):
    train_df = pd.read_csv(f"{path}/train.csv")
    test_df = pd.read_csv(f"{path}/test.csv")
    return train_df, test_df