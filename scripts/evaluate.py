def evaluate(model, X_train, y_train, X_test, y_test):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("train acc:", train_acc)
    print("test acc:", test_acc)