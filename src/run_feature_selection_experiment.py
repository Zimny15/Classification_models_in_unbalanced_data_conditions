import json
import pandas as pd

from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from evaluate import evaluate_model
from results_io import append_results_to_csv
from models import get_baseline_models
from feature_selection import (
    select_features_lasso,
    select_features_random_forest
)


DATASET_NAME = "titanic"
EXPERIMENT_TYPE = "feature_selection"
SEED = 42
TEST_SIZE = 0.2
OUTPUT_FILE = f"results/{DATASET_NAME}_all_results.csv"

FEATURE_SELECTION_METHODS = [
    "lasso",
    "random_forest"
]


def main():
    X, y = load_dataset(DATASET_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
        shuffle=True
    )

    all_results = []

    for method in FEATURE_SELECTION_METHODS:
        print()
        print(f"Feature selection experiment: method = {method}")

        if method == "lasso":
            X_train_selected, X_test_selected, selected_features = select_features_lasso(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                seed=SEED
            )

        elif method == "random_forest":
            X_train_selected, X_test_selected, selected_features = select_features_random_forest(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                seed=SEED,
                max_features=10
            )

        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        print(f"Selected features count: {len(selected_features)}")
        print(f"Selected features: {selected_features}")

        models = get_baseline_models()

        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            model_result = evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train_selected,
                X_test=X_test_selected,
                y_train=y_train,
                y_test=y_test,
                dataset_name=DATASET_NAME,
                experiment_type=f"{EXPERIMENT_TYPE}_{method}",
                seed=SEED
            )

            model_result["feature_selection_method"] = method
            model_result["selected_features_count"] = len(selected_features)
            model_result["selected_features"] = json.dumps(selected_features)

            model_result["best_params"] = json.dumps(
                model.get_params(),
                default=str
            )

            all_results.append(model_result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["feature_selection_method", "roc_auc"],
        ascending=[True, False]
    )

    print()
    print("Feature selection experiment results:")
    print(results_df)

    append_results_to_csv(results_df, OUTPUT_FILE)

    print()
    print(f"Results appended to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()