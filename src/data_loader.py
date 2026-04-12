import pandas as pd

from sklearn.datasets import load_breast_cancer


def load_dataset(dataset_name):
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()

        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        return X, y

    raise ValueError(f"Unsupported dataset name: {dataset_name}")