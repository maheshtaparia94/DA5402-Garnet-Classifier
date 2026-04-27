import io
import os
import mlflow
import pickle
import shutil
import time
import zipfile
import numpy as np
import requests

from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from api.database import Prediction, create_tables, get_db
from api.drift_detector import DriftDetector
from api.schemas import (FeedbackRequest, HealthResponse, PredictResponse,
                         PredictionResult, ReadyResponse)
from src.data.preprocessor import preprocess_spectrum
from src.utils import load_config

load_dotenv()

#Config 
config = load_config()
REGISTRY_NAME = config["mlflow"]["registry_name"]
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:5001")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
APPLY_ADVANCED = config["preprocessing"]["apply_advanced"]
PREDICTIONS_DIR = Path("data/predictions")
LABELED_DIR = Path("data/labeled")

create_tables()
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_VERSION = "production"
DATA_VERSION = "v1"

try:
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(REGISTRY_NAME, "production")
    CURRENT_VERSION = f"v{mv.version}"
    
    # Tags are on the run, not the model version
    run = client.get_run(mv.run_id)
    adv = run.data.tags.get("apply_advanced", "false")
    dv  = run.data.tags.get("data_version", "v1")
    APPLY_ADVANCED = adv.lower() == "true"
    DATA_VERSION = dv
except Exception:
    pass  # keep "production" as default

# threading.Thread(target=_fetch_version, daemon=True).start()

#  Load classes + drift detector at startup
try:
    with open("data/splits/label_encoder.pkl", "rb") as f:
        CLASSES = list(pickle.load(f).classes_)
except Exception:
    CLASSES = config["data"]["classes"]

try:
    DETECTOR = DriftDetector("data/reference/baseline_stats.pkl")
except Exception:
    DETECTOR = None

# Prometheus metrics 
predictions_total = Counter("predictions_total", "Total predictions made")
feedback_total = Counter("feedback_total", "Total feedback submissions")
wrong_total = Counter("wrong_predictions_total","Wrong predictions from feedback")
low_confidence_total = Counter("low_confidence_total", "Low confidence predictions")
actual_error_rate = Gauge("actual_error_rate", "Actual error rate last 24h")
low_confidence_rate = Gauge("low_confidence_rate", "Low confidence rate last 24h")
drift_score = Gauge("drift_score", "Latest drift score")
model_server_up = Gauge("model_server_up", "Model server reachability")
inference_latency = Histogram(
    "inference_latency_ms",
    "End-to-end prediction latency in milliseconds",
    buckets=[10, 25, 50, 100, 200, 500, 1000])

# App
app = FastAPI(title="Garnet Raman Classifier", version="1.0.0")
Instrumentator().instrument(app).expose(app)

def _update_rates(db: Session):
    """Recalculate sliding 24h error and low-confidence rates."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    total = db.query(Prediction).filter(Prediction.timestamp >= cutoff).count()
    if total > 0:
        low = db.query(Prediction).filter(
            Prediction.timestamp >= cutoff,
            Prediction.confidence < CONF_THRESHOLD).count()
        low_confidence_rate.set(low / total)

    labeled = db.query(Prediction).filter(
        Prediction.timestamp >= cutoff,
        Prediction.ground_truth.isnot(None)).count()
    if labeled > 0:
        wrong = db.query(Prediction).filter(
            Prediction.timestamp >= cutoff,
            Prediction.is_wrong == True).count()
        actual_error_rate.set(wrong / labeled)


# Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — always 200 if container is running."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness check — 200 only when model-server is up and model is loaded."""
    try:
        r = requests.get(f"{MODEL_SERVER_URL}/ping", timeout=3)
        if r.status_code == 200:
            model_server_up.set(1)
            return {"status": "ready", "model_version": CURRENT_VERSION,
                    "model_name": REGISTRY_NAME}
    except Exception:
        pass
    model_server_up.set(0)
    raise HTTPException(503, "Model server not ready")

@app.get("/classes")
def get_classes():
    """Return available garnet class names."""
    return {"classes": CLASSES}

@app.get("/predictions/pending")
def get_pending(db: Session = Depends(get_db)):
    """Return predictions that have not yet received ground truth feedback."""
    rows = db.query(Prediction).filter(
        Prediction.ground_truth.is_(None)
    ).order_by(Prediction.timestamp.desc()).all()
    return {"predictions": [
        {
            "id": r.id,
            "filename": r.filename,
            "predicted_class": r.predicted_class,
            "confidence": round(r.confidence, 4),
            "drift_score": round(r.drift_score or 0.0, 4),
            "is_drifted": r.is_drifted,
            "is_low_confidence": r.is_low_confidence,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows ]}


@app.get("/predictions/{prediction_id}/spectrum")
def get_spectrum(prediction_id: int, db: Session = Depends(get_db)):
    """Return raw wavenumber + intensity data for a stored spectrum."""
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(404, "Prediction not found")
    if not record.file_path or not Path(record.file_path).exists():
        raise HTTPException(404, "Spectrum file not found")
    wn, intensity = [], []
    with open(record.file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    wn.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue
    return {"wavenumber": wn, "intensity": intensity}


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), dry_run: bool = False, db: Session = Depends(get_db)):
    """
    Classify one or more Raman spectra.
    Accepts: single .txt file or .zip archive of .txt files.
    """
  
    content = file.file.read()
    files = []

    # ── Parse input
    if file.filename.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    fname = os.path.basename(name)
                    if fname.endswith(".txt") and not fname.startswith((".", "_")):
                        files.append((fname, zf.read(name)))
        except Exception as e:
            raise HTTPException(400, f"Invalid zip file: {e}")
    elif file.filename.endswith(".txt"):
        files.append((file.filename, content))
    else:
        raise HTTPException(400, "Only .txt or .zip files accepted")

    if not files:
        raise HTTPException(400, "No .txt spectrum files found in upload")

    results = []
    try:
        for fname, fcontent in files:

            # Parse spectrum
            wn, intensity = [], []
            for line in fcontent.decode("utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        wn.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue

            if not wn:
                raise HTTPException(422, f"No valid wavenumber/intensity data in {fname}")

            # Preprocess
            X = preprocess_spectrum(
                np.array(wn), np.array(intensity), config, APPLY_ADVANCED).reshape(1, -1).astype(np.float32)

            # Call model-server
            try:
                start = time.time()
                resp = requests.post(
                    f"{MODEL_SERVER_URL}/invocations",
                    json={
                        "dataframe_split": {
                            "columns": [str(i) for i in range(X.shape[1])],
                            "data":    X.tolist()
                        }
                    },
                    timeout=30)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(503, f"Model server error: {e}")

            # Parse probabilities
            raw = resp.json()["predictions"][0]
            probs = np.array(raw, dtype=np.float64)
            if probs.min() < 0:
                # CNN/MLP logits → softmax
                exp   = np.exp(probs - np.max(probs))
                probs = exp / exp.sum()
            else:
                # SVM/RF/PLSDA → normalize
                probs = np.clip(probs, 0, None)
                probs = probs / probs.sum()

            predicted_class = CLASSES[int(np.argmax(probs))]
            confidence = float(np.max(probs))
            is_low = confidence < CONF_THRESHOLD

            # ── Drift detection
            if DETECTOR:
                det_result = DETECTOR.detect(X.flatten())
            else:
                det_result = {"drift_score": 0.0, "is_drifted": False}
            d_score = float(det_result["drift_score"])
            d_flagged = bool(det_result["is_drifted"])
            drift_score.set(d_score)

            # ── Save spectrum file
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_path = PREDICTIONS_DIR / f"{ts}_{fname}"
            with open(file_path, "w") as fout:
                for w, iv in zip(wn, intensity):
                    fout.write(f"{w}\t{iv}\n")

            # Prometheus counters
            predictions_total.inc()
            if is_low:
                low_confidence_total.inc()

            if not dry_run:
                # ── Store to DB 
                db.add(Prediction(
                    filename = fname,
                    file_path = str(file_path),
                    predicted_class = predicted_class,
                    confidence = confidence,
                    is_low_confidence = is_low,
                    drift_score = d_score,
                    is_drifted  = d_flagged,
                    model_version = CURRENT_VERSION,
                    data_version = DATA_VERSION))
                
            results.append(PredictionResult(
                filename = fname,
                predicted_class = predicted_class,
                confidence = confidence,
                probabilities = {c: float(p) for c, p in zip(CLASSES, probs)},
                model_version = CURRENT_VERSION,
                drift_score = d_score))
            
        if not dry_run:
            db.commit()
            _update_rates(db)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    ms = (time.time() - start) * 1000
    inference_latency.observe(ms)

    return PredictResponse(
        predictions = results,
        total = len(results),
        processing_time_ms = ms)


@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit ground truth label for a prediction.
    Flags wrong/uncertain/drifted spectra for retraining.
    """
    record = db.query(Prediction).filter(
        Prediction.filename == req.filename).order_by(Prediction.timestamp.desc()).first()

    if not record:
        raise HTTPException(404, "Prediction not found")

    record.ground_truth = req.ground_truth
    record.is_wrong     = record.predicted_class != req.ground_truth
    db.commit()

    feedback_total.inc()
    if record.is_wrong:
        wrong_total.inc()

    # Move to labeled folder if flagged for retraining
    if (record.is_wrong or record.is_low_confidence or record.is_drifted) \
            and record.file_path and Path(record.file_path).exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = LABELED_DIR / f"{ts}_{req.ground_truth}_{record.filename}"
        shutil.move(record.file_path, dest)
        record.file_path = str(dest)
        db.commit()

    _update_rates(db)
    return {
        "filename": req.filename,
        "predicted_class": record.predicted_class,
        "ground_truth": req.ground_truth,
        "is_wrong": record.is_wrong,
    }


@app.get("/pipeline/runs")
def get_pipeline_runs():
    """Return recent MLflow runs for Streamlit pipeline tab."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(
            config["mlflow"]["experiment_name"])
        if not exp:
            return {"runs": []}
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=100)
        
        # Filter parent runs only (no parentRunId tag)
        parent_runs = [r for r in runs
                       if "mlflow.parentRunId" not in r.data.tags]
        return {"runs": [{
            "model": r.data.tags.get("model_name", r.info.run_name),
            "f1": round(r.data.metrics.get("test_f1_weighted", 0), 4),
            "status": r.info.status,
            "duration": int((r.info.end_time - r.info.start_time)/1000) if r.info.end_time else 0,
            "run_id": r.info.run_id,
        } for r in parent_runs]}
    
    except Exception as e:
        return {"runs": [], "error": str(e)}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())