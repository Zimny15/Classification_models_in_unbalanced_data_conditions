import json
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def build_model_from_trial(model_name, trial, seed):
    if model_name == "Logistic Regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "max_iter": 1000,
            "random_state": seed
        }
        model = LogisticRegression(**params)
        return model, params

    if model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": seed
        }
        model = RandomForestClassifier(**params)
        return model, params

    if model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": seed,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        return model, params

    if model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": seed,
            "verbose": 0
        }
        model = CatBoostClassifier(**params)
        return model, params

    raise ValueError(f"Unsupported model name: {model_name}")


def tune_model(model_name, X_train, y_train, seed, n_trials=20):
    def objective(trial):
        X_train_inner, X_valid_inner, y_train_inner, y_valid_inner = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
            shuffle=True
        )

        model, _ = build_model_from_trial(model_name, trial, seed)
        model.fit(X_train_inner, y_train_inner)

        probabilities = model.predict_proba(X_valid_inner)[:, 1]
        roc_auc = roc_auc_score(y_valid_inner, probabilities)

        return roc_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_validation_score = study.best_value

    return best_params, best_validation_score


def build_final_tuned_model(model_name, best_params, seed):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            **best_params,
            max_iter=1000,
            random_state=seed
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            **best_params,
            random_state=seed
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            **best_params,
            random_state=seed,
            eval_metric="logloss"
        )

    if model_name == "CatBoost":
        return CatBoostClassifier(
            **best_params,
            random_state=seed,
            verbose=0
        )

    raise ValueError(f"Unsupported model name: {model_name}")