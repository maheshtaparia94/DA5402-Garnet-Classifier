# Low-Level Design Document

## 1. API Endpoint Specifications

Base URL: `http://localhost:8000`

### 1.1 Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Model loaded check |
| GET | `/classes` | Available garnet classes |
| POST | `/predict` | Classify spectrum file |
| POST | `/feedback` | Submit ground truth label |
| GET | `/predictions/pending` | Unlabelled predictions |
| GET | `/predictions/{id}/spectrum` | Raw spectrum data |
| GET | `/metrics` | Prometheus metrics |

---

### 1.2 Endpoint Details

**GET /health** → `200 { "status": "ok" }`

**GET /ready** → `200 { "status": "ready", "model_version": "v3", "model_name": "garnet-classifier" }` | `503` model not loaded

**GET /classes** → `200 { "classes": ["Almandine", "Andradite", "Grossular", "Pyrope", "Spessartite"] }`

---

**POST /predict** — Classify one or more spectra

*Request:* `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | `.txt` or `.zip` of `.txt` files |
| dry_run | bool | No | Skip DB write (default: false) |

*File format:* Two-column tab-separated (wavenumber, intensity), no header.

*Response 200:*
```json
{
  "predictions": [{
    "filename": "sample.txt",
    "predicted_class": "Almandine",
    "confidence": 0.934,
    "probabilities": { "Almandine": 0.934, "Andradite": 0.021, "Grossular": 0.018, "Pyrope": 0.015, "Spessartite": 0.012 },
    "drift_score": 0.042,
    "model_version": "v3"
  }],
  "total": 1,
  "processing_time_ms": 799.28
}
```

*Errors:* `400` invalid file type | `400` zip has no .txt | `422` no valid spectrum data | `503` model unavailable

---

**POST /feedback** — Submit ground truth label

*Request:* `application/json` `{ "filename": "sample.txt", "ground_truth": "Almandine" }`

*Response 200:* `{ "filename": "sample.txt", "predicted_class": "Pyrope", "ground_truth": "Almandine", "is_wrong": true }`

*Errors:* `404` prediction not found | `422` missing fields

---

**GET /predictions/pending** — Unlabelled predictions

*Response 200:*
```json
{
  "predictions": [{
    "id": 42, "filename": "sample.txt", "predicted_class": "Pyrope",
    "confidence": 0.61, "is_low_confidence": true, "drift_score": 0.12,
    "timestamp": "2026-04-25T10:30:00Z"
  }]
}
```

---

**GET /predictions/{id}/spectrum** — Raw spectrum data

*Path param:* `id` integer | *Response 200:* `{ "wavenumber": [...], "intensity": [...] }` | `404` not found

---

**GET /metrics** — Prometheus text format, scraped every 15 seconds.

---

## 2. Database Schema

Table: `predictions` (PostgreSQL 15)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PK, auto-increment | Unique ID |
| filename | VARCHAR | NOT NULL | Original filename |
| file_path | VARCHAR | NOT NULL | Stored spectrum path |
| predicted_class | VARCHAR | NOT NULL | Model prediction |
| confidence | FLOAT | NOT NULL | Max probability |
| is_low_confidence | BOOLEAN | NOT NULL | Below threshold |
| drift_score | FLOAT | NOT NULL | Z-score from baseline |
| is_drifted | BOOLEAN | NOT NULL | Drift above 0.3 |
| ground_truth | VARCHAR | NULLABLE | User-submitted label |
| is_wrong | BOOLEAN | NULLABLE | Prediction incorrect |
| model_version | VARCHAR | NOT NULL | MLflow alias at time |
| timestamp | DATETIME | NOT NULL | UTC timestamp |

---

## 3. DVC Pipeline Stages

| Stage | Command | Inputs | Outputs |
|-------|---------|--------|---------|
| preprocess | `python -m src.data.preprocessor` | `data/raw/` | `data/processed/spectra_preprocessed.csv` |
| split_data | `python -m src.data.splitter` | `spectra_preprocessed.csv` | `data/splits/` |
| baseline_stats | `python -m src.data.baseline_stats` | `data/splits/` | `data/reference/` |
| eda | `python -m src.eda.run_eda` | `data/processed/`, `data/splits/` | `reports/eda/` |
| train | `python -m src.models.train_model` | `data/splits/` | `data/pipeline_run_id.txt` |
| evaluate_register | `python -m src.models.evaluate_register` | `pipeline_run_id.txt`, `data/golden/` | [] |

---

## 4. MLflow Tracking Schema

| Category | Key | Example | Description |
|----------|-----|---------|-------------|
| Param | `model_name` | `cnn` | Model selected for this run |
| Param | `cv_folds` | `5` | Cross-validation folds used |
| Param | `hyperparameters` | `lr=0.001` | Hyeperparameters as per model definition |
| Metric | `cv_f1_mean` | `0.912` | Mean CV F1 per hyperparameter trial (child run) |
| Metric | `cv_f1_std` | `0.023` | Std CV F1 per hyperparameter trial (child run) |
| Metric | `test_accuracy` | `0.92` | Test accuracy |
| Metric | `test_f1_weighted` | `0.88` | Weighted F1 |
| Metric | `f1_Almandine` | `0.909` | Per-class F1 |
| Artifact | `confusion_matrix.png` | — | Test confusion matrix |
| Artifact | `shap_summary.png` | — | SHAP feature importance summary |
| Artifact | `precision_recall_curve.png` | — | Precision-recall curve per class |
| Artifact | `sample_prediction.png` | — | Sample spectrum prediction plot |
| Artifact | `classification_report.txt` | — | Full classification report |
| Artifact | `model/` | — | Serialised model |
| Tag | `model_name` | `cnn` | Model type tag |
| Tag | `apply_advanced` | `True` | Preprocessing version — read by FastAPI at startup |
| Tag | `pipeline_run_id` | `367ac4f5` | Links run to its evaluate_register execution |
| Tag | `git_commit` | `23bf72f` | Git SHA at training |

---

## 5. Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `predictions_total` | Counter | Total predictions made |
| `actual_error_rate` | Gauge | Wrong / labeled (24h window) |
| `low_confidence_rate` | Gauge | Low confidence prediction rate |
| `drift_score` | Gauge | Latest Z-score drift value |
| `inference_latency_ms` | Histogram | p50 / p95 / p99 latency |
| `model_server_up` | Gauge | 1 = up, 0 = down |
| `garnet_webapp_requests` | Counter | Requests per Streamlit session (port 8002) |
| `garnet_total_sessions` | Gauge | Total unique browser sessions opened |
