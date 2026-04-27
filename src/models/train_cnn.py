import mlflow
import numpy as np
import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class CNN1D(nn.Module):
    def __init__(self, input_len, n_filters, kernel_size, n_classes, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * 2 * 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes))

    def forward(self, x):
        # x: (batch, features) → (batch, 1, features)
        return self.fc(self.conv(x.unsqueeze(1)))


def _train_one(X_tr, y_tr, X_val, y_val, params, n_classes):
    """Train one CNN config, return val f1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(X_tr.shape[1], params["n_filters"],
                   params["kernel_size"], n_classes,
                   params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.FloatTensor(X_tr).to(device)
    ytr = torch.LongTensor(y_tr).to(device)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True, 
                    generator=torch.Generator().manual_seed(42))

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds =  model(torch.FloatTensor(X_val.astype(np.float32)).
                       to(device)).argmax(dim=1).cpu().numpy()
    return f1_score(y_val, preds, average="weighted", zero_division=0), model


def train(X_train, y_train, n_classes, cv_folds=5):
    """
    Train 1D-CNN with manual grid + 5-fold CV.
    Returns best model, best params, cv_score.
    """
    random.seed(42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search_space = {
        "n_filters": [32, 64, 128],
        "kernel_size": [3, 5, 7],
        "lr": [1e-3, 5e-4, 1e-4],
        "epochs": [50, 80, 100],
        "dropout": [0.2, 0.3, 0.4],
    }

    param_grid = [
        {
            "n_filters": random.choice(search_space["n_filters"]),
            "kernel_size": random.choice(search_space["kernel_size"]),
            "lr": random.choice(search_space["lr"]),
            "epochs": random.choice(search_space["epochs"]),
            "dropout": random.choice(search_space["dropout"]),
        }
        for _ in range(15)]
    
    best_score, best_params = -1, None
    print("LENGTH", len(param_grid))
    for trial, params in enumerate(param_grid):
        scores = []
        for tr, val in cv.split(X_train, y_train):
            score, _ = _train_one(X_train[tr], y_train[tr],
                                  X_train[val], y_train[val],
                                  params, n_classes)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        run_name = f"cnn_f{params['n_filters']}_k{params['kernel_size']}_lr{params['lr']}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("model_name", "cnn")
            mlflow.set_tag("trial_number", str(trial + 1))
            mlflow.log_param("n_filters", params["n_filters"])
            mlflow.log_param("kernel_size", params["kernel_size"])
            mlflow.log_param("lr", params["lr"])
            mlflow.log_param("epochs", params["epochs"])
            mlflow.log_param("dropout", params["dropout"])
            mlflow.log_metric("cv_f1_mean", mean_score)
            mlflow.log_metric("cv_f1_std", std_score)

        if mean_score > best_score:
            best_score  = mean_score
            best_params = params

    # Refit on full X_train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(X_train.shape[1], best_params["n_filters"],
                    best_params["kernel_size"], n_classes,
                    best_params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.FloatTensor(X_train).to(device)
    ytr = torch.LongTensor(y_train).to(device)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)

    model.train()
    for _ in range(best_params["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    return model, best_params, best_score


def predict(model, X):
    """Return class predictions from 1D-CNN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X.astype(np.float32)).
                     to(device)).argmax(dim=1).cpu().numpy()