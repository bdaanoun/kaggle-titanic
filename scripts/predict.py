import pandas as pd

def make_submission(model, test_df, path="../data/test.csv"):
    test_predict = model.predict(test_df)

    submission = pd.DataFrame({
        "PassengerId": pd.read_csv(path)["PassengerId"],
        "Survived": test_predict
    })

    submission.to_csv("../data/submission.csv", index=False)