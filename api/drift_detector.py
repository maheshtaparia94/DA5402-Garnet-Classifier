import pickle
from pathlib import Path

import numpy as np


class DriftDetector:

    def __init__(
        self,
        baseline_path: str = "data/reference/baseline_stats.pkl",
        z_threshold: float = 3.0,
        drift_threshold: float = 0.50):

        with open(baseline_path, "rb") as f:
            stats = pickle.load(f)

        self.mean = np.array(stats["mean"])
        self.std = np.array(stats["std"])
        self.z_threshold = z_threshold
        self.drift_threshold = drift_threshold

        # Avoid division by zero
        self.std = np.where(self.std < 1e-8, 1e-8, self.std)

    def detect(self, X: np.ndarray) -> dict:
        """
        Detect drift in a single preprocessed spectrum.

        Args:
            X: preprocessed spectrum shape (n_features,)

        Returns:
            drift_score: float 0-1 (fraction of drifted features)
            is_drifted:  bool
        """
        z_scores = np.abs((X - self.mean) / self.std)
        drift_score = float(np.mean(z_scores > self.z_threshold))
        is_drifted = drift_score > self.drift_threshold

        return {"drift_score": round(drift_score, 4),
                "is_drifted":  is_drifted}
