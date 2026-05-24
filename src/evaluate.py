import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_roc_auc(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    if len(set(y_test)) == 2:
        return roc_auc_score(y_test, probabilities[:, 1])
    
    return roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro"
    )


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
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_end = time.perf_counter()

    prediction_start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_end = time.perf_counter()

    training_time = train_end - train_start
    prediction_time = prediction_end - prediction_start
    total_modeling_time = training_time + prediction_time

    roc_auc = calculate_roc_auc(model, X_test, y_test)
    
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="binary" if len(set(y_test)) == 2 else "macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="binary" if len(set(y_test)) == 2 else "macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="binary" if len(set(y_test)) == 2 else "macro",
        zero_division=0
    )

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

        "training_time_sec": training_time,
        "prediction_time_sec": prediction_time,
        "total_modeling_time_sec": total_modeling_time
    }

    return results