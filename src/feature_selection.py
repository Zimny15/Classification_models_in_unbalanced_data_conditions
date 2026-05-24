import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def select_features_lasso(X_train, y_train, X_test, seed=42):
    """
    Selekcja zmiennych metodą LASSO.
    Zostawia zmienne, które mają niezerowe współczynniki.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    selector_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        random_state=seed
    )

    selector_model.fit(X_train_scaled, y_train)

    coefficients = selector_model.coef_[0]

    selected_columns = X_train.columns[coefficients != 0]

    if len(selected_columns) == 0:
        selected_columns = X_train.columns

    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    return X_train_selected, X_test_selected, list(selected_columns)


def select_features_random_forest(X_train, y_train, X_test, seed=42, max_features=10):
    """
    Selekcja zmiennych na podstawie ważności cech z Random Forest.
    Zostawia max_features najważniejszych zmiennych.
    """

    selector_model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed
    )

    selector_model.fit(X_train, y_train)

    importances = pd.Series(
        selector_model.feature_importances_,
        index=X_train.columns
    )

    selected_columns = (
        importances
        .sort_values(ascending=False)
        .head(max_features)
        .index
    )

    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    return X_train_selected, X_test_selected, list(selected_columns)