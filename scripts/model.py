from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        max_features="sqrt"
    )

    # Evaluate on split
    clf.fit(X_train, y_train)
    print("train acc:", clf.score(X_train, y_train))
    print("test acc:", clf.score(X_test, y_test))

    clf.fit(X, y)

    return clf

# submission scored 0.78947