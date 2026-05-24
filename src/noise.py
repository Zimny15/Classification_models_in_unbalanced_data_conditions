import numpy as np
import pandas as pd


def add_noise_to_numeric_features(X, noise_level=0.1, seed=42):
    """
    Dodaje szum do kolumn numerycznych.

    noise_level = 0.1 oznacza szum równy 10% odchylenia standardowego danej kolumny.
    """

    X_noisy = X.copy()
    rng = np.random.default_rng(seed)

    numeric_columns = X_noisy.select_dtypes(include=["int64", "float64"]).columns

    for column in numeric_columns:
        std = X_noisy[column].std()

        if std == 0 or pd.isna(std):
            continue

        noise = rng.normal(
            loc=0,
            scale=noise_level * std,
            size=len(X_noisy)
        )

        X_noisy[column] = X_noisy[column] + noise

    return X_noisy

def add_noise_to_categorical_features(X, noise_level=0.1, seed=42):
    """
    Dodaje szum do kolumn kategorycznych.

    noise_level = 0.1 oznacza, że 10% wartości w każdej kolumnie kategorycznej
    zostanie losowo zamienione na inną kategorię z tej samej kolumny.
    """

    X_noisy = X.copy()
    rng = np.random.default_rng(seed)

    categorical_columns = X_noisy.select_dtypes(include=["object", "category", "bool"]).columns

    for column in categorical_columns:
        unique_values = X_noisy[column].dropna().unique()

        if len(unique_values) <= 1:
            continue

        n_rows = len(X_noisy)
        n_noisy = int(noise_level * n_rows)

        if n_noisy == 0:
            continue

        noisy_indices = rng.choice(
            X_noisy.index,
            size=n_noisy,
            replace=False
        )

        for idx in noisy_indices:
            current_value = X_noisy.loc[idx, column]

            possible_values = [
                value for value in unique_values
                if value != current_value
            ]

            if len(possible_values) == 0:
                continue

            X_noisy.loc[idx, column] = rng.choice(possible_values)

    return X_noisy


def add_noise_to_features(X, noise_level=0.1, seed=42):
    """
    Dodaje szum zarówno do zmiennych numerycznych, jak i kategorycznych.
    """

    X_noisy = add_noise_to_numeric_features(
        X=X,
        noise_level=noise_level,
        seed=seed
    )

    X_noisy = add_noise_to_categorical_features(
        X=X_noisy,
        noise_level=noise_level,
        seed=seed
    )

    return X_noisy