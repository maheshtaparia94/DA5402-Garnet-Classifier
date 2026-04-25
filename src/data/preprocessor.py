import numpy as np
import pandas as pd

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pybaselines import Baseline
from src.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config()


def preprocess_spectrum(
    wavenumber: np.ndarray,
    intensity: np.ndarray, 
    config, apply_advanced) -> np.ndarray:
    """
    Preprocess a single Raman spectrum.

    BASIC (always):
      1. Interpolate to standard axis (200-1300, step 1, 1101 pts)
      2. Fill NaN via linear interpolation

    ADVANCED (only if apply_advanced=True):
      3. ARPLS baseline correction
      4. Savitzky-Golay smoothing
      5. Normalization

    Args:
        wavenumber: Raw wavenumber axis (any spacing).
        intensity: Raw intensity values.

    Returns:
        np.ndarray: Preprocessed spectrum of length 1101.
    """
    wv_start = config["wavenumber"]["start"]
    wv_end = config["wavenumber"]["end"]
    n_pts = config["wavenumber"]["n_points"]
    proc = config["preprocessing"]

    # Interpolate to standard axis
    std_axis = np.linspace(wv_start, wv_end, n_pts)
    interp_fn = interp1d(
        wavenumber, intensity,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate")
    spectrum = interp_fn(std_axis)

    # Fill NaN via linear interpolation
    if np.any(np.isnan(spectrum)):
        nan_mask = np.isnan(spectrum)
        idx = np.arange(len(spectrum))
        spectrum[nan_mask] = np.interp(
            idx[nan_mask], idx[~nan_mask], spectrum[~nan_mask])

    if not apply_advanced:
        return spectrum

    # ARPLS baseline correction
    fitter = Baseline()
    baseline, _ = fitter.arpls(
        spectrum,
        lam=proc["baseline"]["lam"],
        max_iter=proc["baseline"]["max_iter"])
    spectrum = spectrum - baseline

    # Savitzky-Golay smoothing
    spectrum = savgol_filter(
        spectrum,
        window_length=proc["smoothing"]["window_length"],
        polyorder=proc["smoothing"]["polyorder"])

    # Normalization
    s_min = spectrum.min()
    s_max = spectrum.max()
    if s_max > s_min:
        spectrum = (spectrum - s_min) / (s_max - s_min)

    return spectrum


def main():
    """
    Main entry point called by Airflow DAG and dvc.yaml.

    Reads all .txt files from data/raw/<class>/
    Applies basic preprocessing (always).
    Applies advanced preprocessing if apply_advanced=true.
    Saves data/processed/spectra_preprocessed.csv.
    Logs data version: v1 or v2.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    out_dir = Path(config["data"]["processed_dir"])
    classes = config["data"]["classes"]
    sep = config["data"]["separator"]

    apply_advanced = config["preprocessing"]["apply_advanced"]
    wv_start = config["wavenumber"]["start"]
    wv_end = config["wavenumber"]["end"]
    n_pts = config["wavenumber"]["n_points"]
    wv_cols = [f"w_{i}" for i in range(wv_start, wv_end + 1)]

    records, n_ok, n_bad = [], 0, 0

    for cls in classes:
        files = sorted((raw_dir / cls).glob("*.txt"))
        logger.info(f"  {cls}: {len(files)} files")

        for fp in files:
            try:
                df = pd.read_csv(
                    fp, sep=sep, header=None,
                    names=["wavenumber", "intensity"])
                wv = df["wavenumber"].values.astype(float)
                it = df["intensity"].values.astype(float)

                spectrum = preprocess_spectrum(wv, it, config, apply_advanced)

                rec = dict(zip(wv_cols, spectrum))
                rec["label"] = cls
                rec["filename"] = fp.name
                records.append(rec)
                n_ok += 1

            except Exception as e:
                logger.error(f"Error {fp.name}: {e}")
                n_bad += 1

    logger.info(f"Processed: {n_ok} ok | {n_bad} errors")

    if n_ok == 0:
        raise ValueError("No spectra processed. Check data/raw/")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)[wv_cols + ["label", "filename"]]
    out = out_dir / "spectra_preprocessed.csv"
    df.to_csv(out, index=False)

    logger.info(f"Saved: {out} | shape: {df.shape}")


if __name__ == "__main__":
    main()