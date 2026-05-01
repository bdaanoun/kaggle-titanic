import pandas as pd

def normalize_title(title):
    if title in ["Mr"]:
        return "Mr"
    elif title in ["Miss", "Mlle", "Ms"]:
        return "Miss"
    elif title in ["Mrs", "Mme"]:
        return "Mrs"
    elif title == "Master":
        return "Master"
    else:
        return "Rare"


def preprocess(train_df, test_df):
    drop_cols = ["PassengerId", "Cabin", "Ticket"]
    train_df.drop(drop_cols, axis=1, inplace=True)
    test_df.drop(drop_cols, axis=1, inplace=True)

    for df in [train_df, test_df]:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["Title"] = df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)

    for df in [train_df, test_df]:
        df["Title"] = df["Title"].apply(normalize_title)

    for df in [train_df, test_df]:
        df["Sex_Pclass"] = df["Sex"].map({"male": 0, "female": 1}) * df["Pclass"]
        
    train_df.drop(["Name", "SibSp", "Parch"], axis=1, inplace=True)
    test_df.drop(["Name", "SibSp", "Parch"], axis=1, inplace=True)

    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

    train_df["Age"] = train_df["Age"].groupby(train_df["Title"]).transform(lambda x: x.fillna(x.median()))
    test_df["Age"] = test_df["Age"].groupby(test_df["Title"]).transform(lambda x: x.fillna(x.median()))

    embarked_mode = train_df["Embarked"].mode()[0]

    # print("age median:", train_df["Age"].isna().sum())

    # for df in [train_df, test_df]:
    #     df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 60, 100], labels=[0,1,2,3])

    # train_df.drop("Age", axis=1, inplace=True)
    # test_df.drop("Age", axis=1, inplace=True)

    train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
    test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

    hot_cols = ["Sex", "Embarked", "Title"]
    train_df = pd.get_dummies(train_df, columns=hot_cols)
    test_df = pd.get_dummies(test_df, columns=hot_cols)

    test_df = test_df.reindex(
        columns=train_df.drop("Survived", axis=1).columns,
        fill_value=0
    )

    return train_df, test_df