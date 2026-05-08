import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(path=None):
    if path is None:
        path = BASE_DIR / "data"
    else:
        path = Path(path)

    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")
    return train_df, test_df