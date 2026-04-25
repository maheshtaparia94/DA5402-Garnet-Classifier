
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.decomposition import PCA
from src.utils import load_config, get_logger

logger  = get_logger(__name__)
config   = load_config()
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]


def save(fig, path):
    """Save figure and close."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def main():
    proc_dir = Path(config["data"]["processed_dir"])
    spl_dir = Path(config["data"]["splits_dir"])
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(proc_dir / "spectra_preprocessed.csv")
    wv_cols = [c for c in df.columns if c.startswith("w_")]
    X = df[wv_cols].values.astype(float)
    labels = df["label"].values
    classes = sorted(df["label"].unique())

    with open(spl_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    wv = np.array([int(c.split("_")[1]) for c in wv_cols])

    # Class Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts  = [np.sum(labels == c) for c in classes]
    bars    = ax.bar(classes, counts, color=COLORS)
    ax.bar_label(bars)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Garnet Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=15)
    save(fig, out_dir / "class_distribution.png")

    # Spectral Overlay
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    fig.delaxes(ax[1][2])
    count = 0
    for cls in classes:
        mask = labels == cls
        ax[count // 3, count % 3].set_title(cls)
        ax[count // 3, count % 3].set_xlabel("Raman Shift (cm-1)")
        ax[count // 3, count % 3].set_ylabel("Intensity (a.u.)")
        for spectrum in X[mask]:
            ax[count // 3, count % 3].plot(wv, spectrum)
        count += 1
    plt.tight_layout()
    save(fig, out_dir / "spectral_overlay.png")


   # ── Plot 4: PCA 2D Scatter ─────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#eaeaf2")
    ax.grid(color="white", linewidth=0.8)
    for i, cls in enumerate(classes):
        mask = labels == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=cls, s=50, zorder=3)
    ax.set_title("Raw Data PCA Score Analysis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", framealpha=0.9)
    save(fig, out_dir / "pca_2d.png")
 
    # ── Plot 5: PCA Explained Variance per Component ──────────────────
    pca_full  = PCA(random_state=42).fit(X)
    var_ratio = pca_full.explained_variance_ratio_
    # show only components with > 1% variance (useful ones)
    useful    = var_ratio[var_ratio > 0.01]
    fig       = plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(useful) + 1), useful)
    plt.ylabel("Explained Variance")
    plt.xlabel("Components")
    plt.ticklabel_format(style="plain")
    save(fig, out_dir / "pca_explained_variance.png")

    # Correlation Heatmap (downsampled)
    step    = max(1, len(wv_cols) // 50)
    X_down  = X[:, ::step]
    wv_down = wv[::step]
    df_corr = pd.DataFrame(X_down, columns=wv_down)
    cormat  = df_corr.corr()
    round(cormat, 2)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(cormat, ax=ax)
    ax.set_title("Wavenumber Correlation Heatmap")
    save(fig, out_dir / "correlation_heatmap.png")
 
    logger.info(f"All EDA plots saved → {out_dir}")

if __name__ == "__main__":
    main()