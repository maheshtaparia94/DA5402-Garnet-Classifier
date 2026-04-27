import mlflow
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score


def train(X_train, y_train, n_classes, cv_folds):
    """
    Train SVM with manual grid over C and kernel.
    Uses 5-fold stratified CV on X_train.
    Returns best model, best params, cv_score.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_grid = [
    {
        "C":      [0.1, 1.0, 10.0],
        "kernel": ["rbf"],
        "gamma":  ["scale", "auto"]
    },
    {
        "C":      [0.1, 1.0, 10.0],
        "kernel": ["linear"],
        "gamma":  ["scale"]
    }]

    best_score, best_params, best_model = -1, None, None
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid, cv=cv,
        scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Log each trial as child run
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(
            run_name=f"svm_C{params['C']}_{params['kernel']}",
            nested=True):
            mlflow.set_tag("model_name",   "svm")
            mlflow.set_tag("trial_number", str(i + 1))
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_mean",
                float(grid.cv_results_['mean_test_score'][i]))
            mlflow.log_metric("cv_f1_std",
                float(grid.cv_results_['std_test_score'][i]))

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    # Refit on full X_train
    best_model = SVC(**best_params, probability=True, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params, best_score


def predict(model, X):
    """Return class predictions."""
    return model.predict(X)