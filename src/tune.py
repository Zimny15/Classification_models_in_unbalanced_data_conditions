import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def calculate_roc_auc(model, X_valid, y_valid):
    probabilities = model.predict_proba(X_valid)

    if len(set(y_valid)) == 2:
        return roc_auc_score(y_valid, probabilities[:, 1])

    return roc_auc_score(
        y_valid,
        probabilities,
        multi_class="ovr",
        average="macro"
    )


def build_model_from_trial(model_name, trial, seed):
    if model_name == "Logistic Regression":
        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter": 2000,
            "random_state": seed
        }
        model = LogisticRegression(**params)
        return model, params

    if model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": seed
        }
        model = RandomForestClassifier(**params)
        return model, params

    if model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        return model, params

    if model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_state": seed,
            "verbose": 0
        }
        model = CatBoostClassifier(**params)
        return model, params

    if model_name == "MLP Perceptron":
        n_layers = trial.suggest_int("n_layers", 1, 3)

        layers = []
        for i in range(n_layers):
            layers.append(
                trial.suggest_int(f"n_units_l{i}", 16, 256)
            )

        params = {
            "n_layers": n_layers,
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "early_stopping": True,
            "validation_fraction": 0.2,
            "max_iter": 1000,
            "random_state": seed
        }

        model_params = params.copy()
        model_params.pop("n_layers")

        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(**model_params)
        )

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

        roc_auc = calculate_roc_auc(
            model=model,
            X_valid=X_valid_inner,
            y_valid=y_valid_inner
        )

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
            solver="liblinear",
            max_iter=2000,
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

    if model_name == "MLP Perceptron":
        n_layers = best_params["n_layers"]

        layers = []
        for i in range(n_layers):
            layers.append(best_params[f"n_units_l{i}"])

        mlp_params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": best_params["activation"],
            "alpha": best_params["alpha"],
            "learning_rate_init": best_params["learning_rate_init"],
            "batch_size": best_params["batch_size"],
            "early_stopping": True,
            "validation_fraction": 0.2,
            "max_iter": 1000,
            "random_state": seed
        }

        return make_pipeline(
            StandardScaler(),
            MLPClassifier(**mlp_params)
        )

    raise ValueError(f"Unsupported model name: {model_name}")