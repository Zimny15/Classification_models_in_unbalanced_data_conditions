import json
import pandas as pd

from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from evaluate import evaluate_model
from tune import tune_model, build_final_tuned_model
from results_io import append_results_to_csv


DATASET_NAME = "breast_cancer"
EXPERIMENT_TYPE = "tuned"
SEED = 42
TEST_SIZE = 0.2
N_TRIALS = 20
OUTPUT_FILE = f"results/{DATASET_NAME}_all_results.csv"

MODELS_TO_TUNE = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "CatBoost"
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

    tuned_results = []

    for model_name in MODELS_TO_TUNE:
        print(f"Tuning model: {model_name}")

        best_params, best_validation_score = tune_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            seed=SEED,
            n_trials=N_TRIALS
        )

        final_model = build_final_tuned_model(
            model_name=model_name,
            best_params=best_params,
            seed=SEED
        )

        model_result = evaluate_model(
            model=final_model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_name=DATASET_NAME,
            experiment_type=EXPERIMENT_TYPE,
            seed=SEED
        )

        model_result["best_validation_score"] = best_validation_score
        model_result["best_params"] = json.dumps(best_params)

        tuned_results.append(model_result)

    tuned_results_df = pd.DataFrame(tuned_results)
    tuned_results_df = tuned_results_df.sort_values(by="roc_auc", ascending=False)

    print()
    print("Tuned experiment results:")
    print(tuned_results_df)

    append_results_to_csv(tuned_results_df, OUTPUT_FILE)

    print()
    print(f"Results appended to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()