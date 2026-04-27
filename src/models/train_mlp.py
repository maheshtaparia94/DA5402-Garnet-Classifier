import mlflow
import numpy as np
import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, n_classes, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_one(X_tr, y_tr, X_val, y_val, params, n_classes):
    """Train one MLP config, return val f1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_tr.shape[1], params["hidden"],
                 n_classes, params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.FloatTensor(X_tr).to(device)
    ytr = torch.LongTensor(y_tr).to(device)
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=16, shuffle=True, 
                    generator=torch.Generator().manual_seed(42))

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xv = torch.FloatTensor(X_val).to(device)
        preds = model(Xv).argmax(dim=1).cpu().numpy()
    return f1_score(y_val, preds, average="weighted", zero_division=0), model


def train(X_train, y_train, n_classes, cv_folds=5):
    """
    Train MLP with manual grid.
    Returns best model, best params, cv_score.
    """
    random.seed(42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search_space = {
        "hidden": [[256, 128], [256, 128, 64], [128, 64], [512, 256], [128]],
        "lr": [1e-3, 5e-4, 1e-4],
        "epochs": [50, 80, 100],
        "dropout": [0.2, 0.3, 0.4],
    }

    param_grid = [{
            "hidden": random.choice(search_space["hidden"]),
            "lr": random.choice(search_space["lr"]),
            "epochs": random.choice(search_space["epochs"]),
            "dropout": random.choice(search_space["dropout"])}
        for _ in range(15)]
    
    best_score, best_params = -1, None
    for trial, params in enumerate(param_grid):
        scores = []
        for tr, val in cv.split(X_train, y_train):
            score, _ = _train_one(X_train[tr], y_train[tr],
                                  X_train[val], y_train[val],
                                  params, n_classes)
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        hidden_str = str(params["hidden"]).replace(" ", "")
        run_name = f"mlp_h{hidden_str}_lr{params['lr']}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("model_name", "mlp")
            mlflow.set_tag("trial_number", str(trial + 1))
            mlflow.log_param("hidden", str(params["hidden"]))
            mlflow.log_param("lr", params["lr"])
            mlflow.log_param("epochs", params["epochs"])
            mlflow.log_param("dropout", params["dropout"])
            mlflow.log_metric("cv_f1_mean", mean_score)
            mlflow.log_metric("cv_f1_std", std_score)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit best config on full X_train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_train.shape[1], best_params["hidden"],
                 n_classes, best_params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.FloatTensor(X_train).to(device)
    ytr = torch.LongTensor(y_train).to(device)
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model.train()
    for _ in range(best_params["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    return model, best_params, best_score


def predict(model, X):
    """Return class predictions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X.astype(np.float32)).to(device))
        return preds.argmax(dim=1).cpu().numpy()
