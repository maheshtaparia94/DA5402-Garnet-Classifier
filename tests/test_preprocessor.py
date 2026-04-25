"""
test_preprocessor.py - Unit tests for preprocessing and drift detection.

5 unit tests:
  1. Output shape is always 1101
  2. NaN values are handled — no NaN in output
  3. Empty spectrum raises error
  4. Basic vs advanced preprocessing produce different results
  5. Drift detector returns expected keys
"""

import numpy as np
import pytest

from src.data.preprocessor import preprocess_spectrum
from src.utils import load_config

config = load_config()


def make_spectrum(n=500):
    """Create a valid Raman spectrum with n points in 200-1300 range."""
    wn = np.linspace(200, 1300, n)
    intensity = np.random.rand(n) * 1000
    return wn, intensity


#  Unit Test 1
def test_output_shape():
    """Output must always be 1101 points — fixed model input shape."""
    wn, intensity = make_spectrum()
    out = preprocess_spectrum(wn, intensity, config, False)
    assert out.shape == (1101,), f"Expected (1101,) got {out.shape}"


# Unit Test 2
def test_nan_handled():
    """NaN values must be interpolated — never reach the model."""
    wn, intensity = make_spectrum()
    intensity[10:20] = np.nan
    out = preprocess_spectrum(wn, intensity, config, False)
    assert not np.isnan(out).any(), "NaN found in preprocessed output"


# Unit Test 3
def test_empty_spectrum_raises():
    """Empty spectrum must raise an error — not silently return zeros."""
    with pytest.raises(Exception):
        preprocess_spectrum(np.array([]), np.array([]), config, False)


# Test 4 
def test_basic_vs_advanced_differ():
    """Basic and advanced preprocessing must produce different outputs."""
    wn, intensity = make_spectrum()
    basic = preprocess_spectrum(wn, intensity, config, False)
    advanced = preprocess_spectrum(wn, intensity, config, True)
    assert not np.array_equal(basic, advanced), \
        "Basic and advanced outputs are identical — check ARPLS baseline"


# Test 5
def test_drift_detector_output_keys():
    """DriftDetector must return drift_score and is_drifted keys."""
    from api.drift_detector import DriftDetector
    detector = DriftDetector("data/reference/baseline_stats.pkl")
    wn, intensity = make_spectrum()
    X = preprocess_spectrum(wn, intensity, config, False)
    result = detector.detect(X)
    assert "drift_score" in result, "Missing drift_score key"
    assert "is_drifted"  in result, "Missing is_drifted key"
    assert isinstance(result["drift_score"], float)
    assert isinstance(result["is_drifted"],  bool)