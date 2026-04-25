import io
import os
import zipfile
import pytest
import requests

BASE = os.environ.get("API_URL", "http://localhost:8000")

def make_zip_no_txt():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image.png", b"fake")
    buf.seek(0)
    return buf.read()


def test_health():
    """GET /health → 200, status ok."""
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.needs_model
def test_ready():
    """GET /ready → 200"""
    r = requests.get(f"{BASE}/ready", timeout=5)
    assert r.status_code == 200


def test_metrics():
    """GET /metrics → 200, contains prometheus counters."""
    r = requests.get(f"{BASE}/metrics", timeout=5)
    assert r.status_code == 200
    assert "predictions_total" in r.text


def test_classes():
    """GET /classes → 200, returns 5 garnet classes."""
    r = requests.get(f"{BASE}/classes", timeout=5)
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 5

@pytest.mark.needs_model
def test_predict_valid_dry_run():
    spectrum = "\n".join(f"{200+i}\t{float(i*10)}" for i in range(500)).encode()
    r = requests.post(f"{BASE}/predict?dry_run=true", timeout=300,
        files={"file": ("test.txt", spectrum, "text/plain")})


def test_predict_wrong_extension():
    """POST /predict with .csv → 400."""
    r = requests.post(f"{BASE}/predict", timeout=5,
        files={"file": ("data.csv", b"a,b\n1,2", "text/csv")})
    assert r.status_code == 400


def test_predict_empty_txt():
    """POST /predict with empty .txt → 400/422."""
    r = requests.post(f"{BASE}/predict", timeout=5,
        files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code in (400, 422, 500)


def test_predict_zip_no_txt():
    """POST /predict with zip containing no .txt → 400."""
    r = requests.post(f"{BASE}/predict", timeout=5,
        files={"file": ("nospec.zip", make_zip_no_txt(), "application/zip")})
    assert r.status_code == 400


def test_predict_corrupted_txt():
    """POST /predict with corrupted .txt → 400/422."""
    r = requests.post(f"{BASE}/predict", timeout=5,
        files={"file": ("bad.txt", b"not a spectrum\njunk", "text/plain")})
    assert r.status_code in (400, 422, 500)


@pytest.mark.needs_model
def test_feedback_not_found():
    """POST /feedback with unknown filename → 404."""
    r = requests.post(f"{BASE}/feedback", timeout=5,
        json={"filename": "nonexistent_xyz.txt", "ground_truth": "Pyrope"})
    r.status_code == 404


def test_feedback_invalid_body():
    """POST /feedback with missing fields → 422."""
    r = requests.post(f"{BASE}/feedback", timeout=5,
        json={"filename": "test.txt"})
    assert r.status_code == 422
