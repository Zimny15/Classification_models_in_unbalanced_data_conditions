import pandas as pd
from pathlib import Path

from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_breast_cancer_dataset():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    return X, y


def load_titanic_dataset():
    file_path = DATA_DIR / "titanic_agh.csv"
    df = pd.read_csv(file_path)

    df = df.drop(columns=["ID_pasażera"])

    df["Płeć"] = df["Płeć"].map({"male": 0, "female": 1})
    df["Port_zaokrętowania"] = df["Port_zaokrętowania"].map({"S": 0, "C": 1, "Q": 2})

    df = df.dropna(subset=["Port_zaokrętowania"])

    X = df.drop(columns=["Przeżył"]).copy()
    y = df["Przeżył"].copy()

    return X, y


def load_dataset(dataset_name):
    if dataset_name == "breast_cancer":
        return load_breast_cancer_dataset()

    if dataset_name == "titanic":
        return load_titanic_dataset()

    raise ValueError(f"Unsupported dataset name: {dataset_name}")