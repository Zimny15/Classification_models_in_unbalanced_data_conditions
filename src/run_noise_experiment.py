import json

import pandas as pd

from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from evaluate import evaluate_model
from results_io import append_results_to_csv
from models import get_baseline_models
from noise import add_noise_to_features


DATASET_NAME = "titanic"
EXPERIMENT_TYPE = "noise"
SEED = 42
TEST_SIZE = 0.2
OUTPUT_FILE = f"results/{DATASET_NAME}_all_results.csv"

NOISE_LEVELS = [
    0.05,
    0.10,
    0.20,
    0.30
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

    for noise_level in NOISE_LEVELS:
        print()
        print(f"Noise experiment: noise level = {noise_level}")

        X_train_noisy = add_noise_to_features(
            X=X_train,
            noise_level=noise_level,
            seed=SEED
        )

        models = get_baseline_models()

        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            model_result = evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train_noisy,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                dataset_name=DATASET_NAME,
                experiment_type=f"{EXPERIMENT_TYPE}_{int(noise_level * 100)}",
                seed=SEED
            )

            model_result["noise_level"] = noise_level

            model_result["best_params"] = json.dumps(
                model.get_params(),
                default=str
            )
            
            all_results.append(model_result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["noise_level", "roc_auc"],
        ascending=[True, False]
    )

    print()
    print("Noise experiment results:")
    print(results_df)

    append_results_to_csv(results_df, OUTPUT_FILE)

    print()
    print(f"Results appended to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()