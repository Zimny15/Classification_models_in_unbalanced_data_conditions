import pandas as pd
import json

from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from models import get_baseline_models
from evaluate import evaluate_model
from results_io import append_results_to_csv


DATASET_NAME = "titanic"
EXPERIMENT_TYPE = "baseline"
SEED = 42
TEST_SIZE = 0.2
OUTPUT_FILE = f"results/{DATASET_NAME}_all_results.csv"


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

    baseline_models = get_baseline_models()
    baseline_results = []

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Number of all samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print()

    for model_name, model in baseline_models.items():
        print(f"Training and evaluating: {model_name}")

        model_results = evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_name=DATASET_NAME,
            experiment_type=EXPERIMENT_TYPE,
            seed=SEED
        )

        model_results["best_params"] = json.dumps(model.get_params(), default=str)

        baseline_results.append(model_results)

    baseline_results_df = pd.DataFrame(baseline_results)
    baseline_results_df = baseline_results_df.sort_values(by="roc_auc", ascending=False)

    print()
    print("Baseline experiment results:")
    print(baseline_results_df)

    append_results_to_csv(baseline_results_df, OUTPUT_FILE)
    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()