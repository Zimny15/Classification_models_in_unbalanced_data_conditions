import pandas as pd
import json

from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from evaluate import evaluate_model
from results_io import append_results_to_csv
from balance import create_imbalanced_dataset, get_class_balance_info
from models import get_baseline_models


DATASET_NAME = "titanic"
EXPERIMENT_TYPE = "class_balance"
SEED = 42
TEST_SIZE = 0.2
OUTPUT_FILE = f"results/{DATASET_NAME}_all_results.csv"

POSITIVE_CLASS = 1

CLASS_RATIOS = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
    0.01
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

    for positive_ratio in CLASS_RATIOS:
        print()
        print(f"Class balance experiment: positive class ratio = {positive_ratio}")

        X_train_balanced, y_train_balanced = create_imbalanced_dataset(
            X=X_train,
            y=y_train,
            positive_class=POSITIVE_CLASS,
            positive_ratio=positive_ratio,
            seed=SEED
        )

        balance_info = get_class_balance_info(
            y=y_train_balanced,
            positive_class=POSITIVE_CLASS
        )

        print(y_train_balanced.value_counts(normalize=True))

        models = get_baseline_models()

        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            model_result = evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train_balanced,
                X_test=X_test,
                y_train=y_train_balanced,
                y_test=y_test,
                dataset_name=DATASET_NAME,
                experiment_type=f"{EXPERIMENT_TYPE}_{int(positive_ratio * 100)}",
                seed=SEED
            )

            model_result["positive_class_ratio"] = balance_info["positive_class_ratio"]
            model_result["positive_class_count"] = balance_info["positive_class_count"]
            model_result["train_size_after_balancing"] = balance_info["train_size_after_balancing"]

            model_result["best_params"] = json.dumps(
                model.get_params(),
                default=str
            )

            all_results.append(model_result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["positive_class_ratio", "roc_auc"],
        ascending=[False, False]
    )

    print()
    print("Class balance experiment results:")
    print(results_df)

    append_results_to_csv(results_df, OUTPUT_FILE)

    print()
    print(f"Results appended to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()