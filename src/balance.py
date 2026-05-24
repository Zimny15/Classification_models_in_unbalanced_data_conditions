import pandas as pd


def create_imbalanced_dataset(X, y, positive_class=1, positive_ratio=0.5, seed=42):
    """
    Tworzy zbiór treningowy z określonym udziałem klasy pozytywnej.

    positive_ratio = 0.5 oznacza 50% klasy pozytywnej
    positive_ratio = 0.1 oznacza 10% klasy pozytywnej
    """

    data = X.copy()
    data["target"] = y.values

    positive_data = data[data["target"] == positive_class]
    negative_data = data[data["target"] != positive_class]

    n_positive_available = len(positive_data)
    n_negative_available = len(negative_data)

    # Liczymy, ile przykładów klasy negatywnej potrzeba dla danego balansu
    n_negative_needed = int(n_positive_available * (1 - positive_ratio) / positive_ratio)

    # Jeśli potrzebujemy więcej negatywnych niż mamy, zmniejszamy liczbę pozytywnych
    if n_negative_needed > n_negative_available:
        n_negative_needed = n_negative_available
        n_positive_needed = int(n_negative_needed * positive_ratio / (1 - positive_ratio))
    else:
        n_positive_needed = n_positive_available

    positive_sample = positive_data.sample(
        n=n_positive_needed,
        random_state=seed
    )

    negative_sample = negative_data.sample(
        n=n_negative_needed,
        random_state=seed
    )

    balanced_data = pd.concat([positive_sample, negative_sample])
    balanced_data = balanced_data.sample(frac=1, random_state=seed)

    X_balanced = balanced_data.drop(columns=["target"])
    y_balanced = balanced_data["target"]

    return X_balanced, y_balanced

def get_class_balance_info(y, positive_class=1):
    positive_count = (y == positive_class).sum()
    total_count = len(y)
    positive_ratio = positive_count / total_count

    return {
        "positive_class_ratio": positive_ratio,
        "positive_class_count": positive_count,
        "train_size_after_balancing": total_count
    }