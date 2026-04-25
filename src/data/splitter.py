import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config()


def main():
    proc_dir = Path(config["data"]["processed_dir"])
    out_dir = Path(config["data"]["splits_dir"])
    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size"]
    seed = config["split"]["random_state"]

    df = pd.read_csv(proc_dir / "spectra_preprocessed.csv")
    wv_cols = [c for c in df.columns if c.startswith("w_")]
    X = df[wv_cols].values.astype(float)
    labels = df["label"].values
    n_cls = len(np.unique(labels))

    le = LabelEncoder()
    y = le.fit_transform(labels)
    logger.info(f"Classes: {list(le.classes_)} | Total: {len(y)}")

    # Ensure test set has at least 1 sample per class
    min_test = n_cls / len(y)
    test_size = max(test_size, min_test)

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y)

    # Ensure val set has at least 1 sample per class
    min_val = n_cls / len(idx_trainval)
    val_ratio = max(val_size / (1.0 - test_size), min_val)

    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio,
        random_state=seed, stratify=y[idx_trainval])

    logger.info(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X[idx_train])
    np.save(out_dir / "X_val.npy",   X[idx_val])
    np.save(out_dir / "X_test.npy",  X[idx_test])
    np.save(out_dir / "y_train.npy", y[idx_train])
    np.save(out_dir / "y_val.npy",   y[idx_val])
    np.save(out_dir / "y_test.npy",  y[idx_test])

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    logger.info(f"Splits saved → {out_dir}")


if __name__ == "__main__":
    main()