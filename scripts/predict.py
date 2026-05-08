import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def make_submission(model, test_df, passenger_ids):
    test_predict = model.predict(test_df)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": test_predict
    })

    submission.to_csv(BASE_DIR / "data" / "submission.csv", index=False)