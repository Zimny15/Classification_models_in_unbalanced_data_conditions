from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name,
    experiment_type,
    seed
):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    n_features = X_train.shape[1]

    results = {
        "dataset_name": dataset_name,
        "model": model_name,
        "experiment_type": experiment_type,
        "seed": seed,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_features": n_features,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_validation_score": None,
        "best_params": None
    }

    return results