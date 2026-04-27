import mlflow
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score


def train(X_train, y_train, n_classes, cv_folds):
    """
    Train RF with manual grid.
    Uses 5-fold stratified CV on X_train.
    Returns best model, best params, cv_score.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5]}
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=cv,
        scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)

    for i, params in enumerate(grid.cv_results_['params']):
        run_name = f"rf_n{params['n_estimators']}_d{params['max_depth']}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("model_name",   "rf")
            mlflow.set_tag("trial_number", str(i + 1))
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_mean",
                float(grid.cv_results_['mean_test_score'][i]))
            mlflow.log_metric("cv_f1_std",
                float(grid.cv_results_['std_test_score'][i]))

    best_params = grid.best_params_
    best_score  = grid.best_score_

    # Refit on full X_train
    best_model = RandomForestClassifier(**best_params,
                                        random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    return best_model, best_params, best_score


def predict(model, X):
    """Return class predictions."""
    return model.predict(X)