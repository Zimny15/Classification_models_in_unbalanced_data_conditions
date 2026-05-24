from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_baseline_models():
    baseline_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "MLP Perceptron": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42
            )
        )
    }

    return baseline_models
