
import mlflow
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score


def train(X_train, y_train, n_classes, cv_folds):
    """
    Train PLS-DA with GridSearch over n_components.
    Uses 5-fold stratified CV on X_train.
    Returns best model, best params, cv_scores.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_grid = {"n_components": [2, 3, 5, 8, 10, 15]}

    best_score, best_n, best_model = -1, None, None

    for trial, n in enumerate(param_grid["n_components"]):        
        # PLS needs one-hot Y
        Y = label_binarize(y_train, classes=np.arange(n_classes))
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        model = PLSRegression(n_components=min(n, X_train.shape[1]))

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            Ytr, yval = Y[train_idx], y_train[val_idx]
            model.fit(Xtr, Ytr)
            preds = np.argmax(model.predict(Xval), axis=1)
            scores.append(f1_score(yval, preds, average="weighted",
                                   zero_division=0))

        mean_score = np.mean(scores)
        std_score  = np.std(scores)
 
        # one per hyperparameter trial
        with mlflow.start_run(run_name=f"plsda_n{n}", nested=True):
            mlflow.set_tag("model_name",   "plsda")
            mlflow.set_tag("trial_number", str(trial + 1))
            mlflow.log_param("n_components", n)
            mlflow.log_metric("cv_f1_mean",  mean_score)
            mlflow.log_metric("cv_f1_std",   std_score)

        if mean_score > best_score:
            best_score = mean_score
            best_n = n
            best_model = model

    # Refit best model on full X_train
    Y_full = label_binarize(y_train, classes=np.arange(n_classes))
    if Y_full.shape[1] == 1:
        Y_full = np.hstack([1 - Y_full, Y_full])

    best_model = PLSRegression(
        n_components=min(best_n, X_train.shape[1]))
    best_model.fit(X_train, Y_full)

    return best_model, {"n_components": best_n}, best_score


def predict(model, X):
    """Return logits."""
    return model.predict(X)