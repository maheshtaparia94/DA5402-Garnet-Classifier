import json
import pickle
import numpy as np

from pathlib import Path
from src.utils import load_config, get_logger

logger = get_logger(__name__)
config    = load_config()

def compute_stats(X_train, std_threshold):
    """Compute mean, std per feature from training data."""
    return {
        "mean": X_train.mean(axis=0).tolist(),
        "std": X_train.std(axis=0).tolist(),
        "std_threshold": std_threshold,
        "n_train": int(len(X_train)),
        "n_features": int(X_train.shape[1])
    }


def main():
    """Main entry point."""
    splits = Path(config["data"]["splits_dir"])
    ref_dir = Path(config["data"]["reference_dir"])
    threshold = config["baseline_stats"]["std_threshold"]

    X_train = np.load(splits / "X_train.npy")
    stats   = compute_stats(X_train, threshold)

    ref_dir.mkdir(parents=True, exist_ok=True)

    with open(ref_dir / "baseline_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    with open(ref_dir / "baseline_stats.json", "w") as f:
        json.dump(stats, f)

    logger.info(f"Baseline saved → {ref_dir} | n_train={len(X_train)}")


if __name__ == "__main__":
    main()