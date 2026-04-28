# High-Level Design Document

---

## 1. Problem Statement

Geologists classify garnet specimens by measuring their Raman spectra. Manual classification requires expert knowledge and is time consuming. This system automates classification of 5 garnet mineral types from raw spectrometer output with real-time confidence scores and a continuous improvement loop.

**Input:** Raw Raman spectrum `.txt` file (wavenumber vs intensity, tab-separated)
**Output:** Predicted class, confidence score, drift score, probabilities
**Classes:** Almandine · Andradite · Grossular · Pyrope · Spessartite

---

## 2. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| UI | Streamlit | 1.45.0 |
| API Gateway | FastAPI + Uvicorn | 0.111.0 |
| Model Serving | MLflow Models Serve | 3.11.1 |
| ML Models | scikit-learn, PyTorch | 1.8.0, 2.11.0 |
| Experiment Tracking | MLflow | 3.11.1 |
| Pipeline Orchestration | Apache Airflow | 2.9.3 |
| Data Versioning | DVC | 3.67.1 |
| Predictions DB | PostgreSQL | 15 |
| Monitoring | Prometheus + Grafana | latest |
| CI/CD | GitHub Actions (self-hosted) | — |
| Containerisation | Docker Compose | v2 |
| Explainability | SHAP | 0.51.0 |

---

## 3. Design Choices and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Serving strategy | Online real-time | Lab instruments produce spectra and used that spectra for classification |
| Inference latency | CNN on CPU ~900ms | Acceptable for lab use — not real-time streaming. GPU deployment reduces to < 200ms |
| Model training | One model from `plsda, svm, rf, mlp, 1D-cnn are selected via `params.yaml` | Try different models from classical(plsda), advacned ML(svm, rf) and DL(mlp, cnn) for comparsion |
| Hyperparameter search | GridSearchCV (SVM/RF), random search (MLP/CNN) | Each trial logged as MLflow child run — best config selected by CV F1 score |
| Model registry | MLflow aliases | MLflow stages deprecated since v2.9 — aliases are the official approach |
| API + model separation | FastAPI + model-server | Decouples ML dependencies from API code — each replaceable independently |
| CI/CD runner | Self-hosted | Since everything is local we prefer self hosted but for production this can be moved to github environment and for deployment we use ssh to connect live server |
| Data versioning | DVC + git | Code and data versioned together — full reproducibility at any commit |
| Drift detection | Z-score vs baseline | Lightweight, interpretable, no external service needed |
| Retraining trigger | git commit from Airflow | Decouples training from deployment — CI/CD runs when new model is registered |

---

## 4. Assumptions

| Assumption | Rationale |
|-----------|-----------|
| Golden test set (15%) is fixed and never used in training | Provides a constant, unbiased evaluation benchmark — new model must beat current @production on this set |
| `@production` alias always points to the best deployed model | MLflow aliases survive restarts and code changes — more robust than file-based tracking |
| `@production_previous` enables instant rollback | Keeping one version back is sufficient for our deployment frequency |
| Online serving is appropriate | Lab workflow produces individual spectra — real-time response required |
| Self-hosted runner has all dependencies pre-installed | Avoids slow installs on every CI run — runner machine is the production machine |
| Z-score threshold 0.5 indicates drift | Empirically chosen from training distribution — tunable via `params.yaml` |
| Confidence threshold 0.70 for low-confidence flag | Below 70% confidence warrants expert review |
| DVC golden set never changes after creation | Golden set created once, frozen — ensures evaluation consistency across versions |

---

## 5. Docker Compose Structure

The system is split into three Docker Compose files for separation of concerns and independent startup:

| File | Services | Purpose |
|------|----------|---------|
| `docker-compose.mlflow.yml` | mlflow-server, mlflow-postgres | MLflow tracking server and its database — started first as other services depend on it |
| `docker-compose.airflow.yml` | airflow-webserver, airflow-scheduler, airflow-triggerer, airflow-init, airflow-postgres | Pipeline orchestration — independent from serving stack |
| `docker-compose.yml` | FastAPI, model-server, app-postgres, Prometheus, Grafana, Alertmanager, node-exporter, Streamlit | Complete application stack — serving, monitoring and UI |

**Startup order:**
```
1. docker compose -f docker-compose.mlflow.yml up -d
2. docker compose -f docker-compose.airflow.yml up -d
3. docker compose up -d
```

---

## 6. Loose Coupling Design

```
Streamlit ──REST──▶ FastAPI ──REST──▶ model-server
   (UI)            (gateway)         (ML inference)
```

- Streamlit: zero ML code only HTTP calls
- FastAPI: zero model loading only HTTP to model-server
- model-server: standalone MLflow service no FastAPI dependency
- The codebase follows the functional programming paradigm — pure functions with no shared mutable state. Python PEP8 coding style is followed throughout.

---

## 7. Security

- All inter-service communication on private Docker networks
- No service exposed beyond UI (8501) and API (8000) ports
- Database credentials in `.env` — excluded from git
- Alertmanager email credentials stored as placeholders
- **Future:** JWT authentication on FastAPI endpoints for more secure system

---

## 8. Scalability & Future Growth

The system is architected for high scalability, ensuring that as demands increase, the infrastructure can expand without requiring modifications to the core application code.

| Component | Scaling Mechanism |
| :--- | :--- |
| **FastAPI** | Supports **horizontal scaling** by deploying multiple Uvicorn workers or additional container instances. |
| **Model Serving** | The existing server is designed for modularity, allowing for a seamless transition to **TorchServe** when GPU-accelerated scaling is required. |
| **MLflow** | The tracking architecture natively supports **S3** or **Azure Blob Storage**, providing high availability and durability for large-scale model artifacts. |
| **DVC** | Integration ensures seamless scaling of datasets by leveraging remote storage backends, keeping the local environment lightweight. |

