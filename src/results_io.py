import os

RESULT_COLUMNS = [
    "dataset_name",
    "model",
    "experiment_type",
    "seed",

    "n_train_samples",
    "n_test_samples",
    "n_features",

    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",

    "training_time_sec",
    "prediction_time_sec",
    "total_modeling_time_sec",

    "best_validation_score",
    "best_params",

    "positive_class_ratio",
    "train_size_after_balancing",

    "noise_level",

    "feature_selection_method",
    "selected_features_count",
    "selected_features",

    "ensemble_models",
    "ensemble_source_experiments"
]


def append_results_to_csv(results_df, output_file):
    for column in RESULT_COLUMNS:
        if column not in results_df.columns:
            results_df[column] = None

    results_df = results_df[RESULT_COLUMNS]

    file_exists = os.path.exists(output_file)

    results_df.to_csv(
        output_file,
        mode="a",
        header=not file_exists,
        index=False
    )