import mlflow
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def train(X_train, y_train, n_classes):
    """
    Train SVM with manual grid over C and kernel.
    Uses 5-fold stratified CV on X_train.
    Returns best model, best params, cv_score.
    """
    n_splits = max(2, min(5, int(np.bincount(y_train).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    param_grid = [{"C": 0.1, "kernel": "rbf", "gamma": "scale"},
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        {"C": 10.0,"kernel": "rbf", "gamma": "scale"},
        {"C": 10.0, "kernel": "linear", "gamma": "scale"}]

    best_score, best_params, best_model = -1, None, None

    for trial, params in enumerate(param_grid):
        model  = SVC(**params, probability=True, random_state=42)
        scores = []
        for tr, val in cv.split(X_train, y_train):
            model.fit(X_train[tr], y_train[tr])
            preds = model.predict(X_train[val])
            scores.append(f1_score(y_train[val], preds,
                                   average="weighted", zero_division=0))
            
        mean_score = np.mean(scores)
        std_score  = np.std(scores)
        with mlflow.start_run( run_name=f"svm_C{params['C']}_{params['kernel']}",
            nested=True):
            mlflow.set_tag("model_name",   "svm")
            mlflow.set_tag("trial_number", str(trial + 1))
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_mean", mean_score)
            mlflow.log_metric("cv_f1_std",  std_score)
            
        if mean_score > best_score:
            best_score  = mean_score
            best_params = params
            best_model  = model

    # Refit on full X_train
    best_model = SVC(**best_params, probability=True, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params, best_score


def predict(model, X):
    """Return class predictions."""
    return model.predict(X)