# Garnet Raman Classifier

End-to-end MLOps system for automated classification of garnet mineral Raman spectra into 5 types: **Almandine, Andradite, Grossular, Pyrope, Spessartite**.

---

## 1. Project Structure

```
.
│
├── api/        
│   ├── main.py                   
│   ├── database.py              
│   ├── drift_detector.py        
│   └── schemas.py               
│
├── airflow/
│   └── dags/
│       └── garnet_pipeline_dag.py
│
├── data/
│   ├── raw.dvc       
│   ├── golden.dvc 
│   └── model_registry_log.txt   
│
├── docker/
│   ├── Dockerfile.airflow      
│   ├── Dockerfile.app          
│   ├── Dockerfile.mlflow 
│   ├── Dockerfile.model-server 
│   └── Dockerfile.fastapi           
│
├── docs/
│   ├── ARCHITECTURE.md          
│   ├── HLD.md                   
│   ├── LLD.md                    
│   ├── TEST_PLAN.md              
│   ├── USER_MANUAL.md
│   └── Architecture_Diagram.png
│ 
├── .github/
│   └── workflows/
│       └── ci_model_deployment.yml
│
├── monitoring/
│   ├── alertmanager.yml     
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules.yml
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/ 
│           │   ├── dashboard.json
│           │   └── dashboard.yml
│           └── datasources/
│               └── datasource.yml
│
├── requirements/
│   ├── requirements-app.txt
│   ├── requirements-mlflow.txt
│   ├── requirements-model.txt
│   ├── requirements-fastpai.txt
│   └── requirements-train.txt
│
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   ├── splitter.py        
│   │   ├── baseline_stats.py 
│   │   └── eda.py          
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── train_svm.py
│   │   ├── train_rf.py
│   │   ├── train_plsda.py
│   │   ├── train_mlp.py
│   │   ├── train_cnn.py
│   │   └── evaluate_register.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocessor.py 
│   └── test_api.py            
│
├── webapp/
│   └── app.py             
│
├── docker-compose.yml
├── docker-compose.mlflow.yml 
├── docker-compose.airflow.yml
├── dvc.lock
├── .dvcignore
├── dvc.yaml
├── ,gitignore
├── params.yaml                 
├── MLproject  
├── pyproject.toml         
├── python_env.yaml
├── readme.md
└── requirements.txt
```

---

## 2. Setup and Installation

### Prerequisites

- Python 3.12
- Docker + Docker Compose v2
- Git + DVC
- 8 GB RAM minimum

### Installation

```bash
# 1. Clone repository
git clone https://github.com/maheshtaparia94/DA5402-Garnet-Classifier.git
cd DA5402-Garnet-Classifier

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with the following variables and update as per your need:
AIRFLOW_UID=1000
AIRFLOW_PROJ_DIR=.
PYTHONPATH=/opt/airflow/project
MLFLOW_TRACKING_URI=http://localhost:5000
PSQL_MLFLOW_USER=mlflow
PSQL_MLFLOW_PASSWORD=mlflow
PSQL_MLFLOW_DB=mlflow
PSQL_APP_USER=garnet
PSQL_APP_PASSWORD=garnet
PSQL_APP_DB=garnet
PSQL_APP_HOST=localhost
MAILTRAP_USERNAME=your_email@gmail.com
MAILTRAP_PASSWORD=your_app_password
```

### Running the System

```bash
# 1. MLflow — tracking server and model registry
docker compose -f docker-compose.mlflow.yml up -d

# 2. Airflow — pipeline orchestration
docker compose -f docker-compose.airflow.yml up -d

# 3. Application — serving, monitoring and UI
docker compose up -d

# Verify all running
docker ps
```

### Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit UI | http://localhost:8501 | — |
| FastAPI Docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | airflow / airflow |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### Running Tests

```bash
# Unit tests
python -m pytest tests/test_preprocessor.py -v

# Integration tests
API_URL=http://localhost:8000 python -m pytest tests/test_api.py -v
```

### GitHub Actions Self-Hosted Runner

```bash
# Go to: GitHub repo → Settings → Actions → Runners → New self-hosted runner
# Follow instructions, then:
./run.sh
```

---

## 3. Implementation and Assumptions

### Implementation

- **Single model per run** — `training.model_name` in `params.yaml` controls which model trains
- **Hyperparameter search** — GridSearchCV for SVM/RF, random search for MLP/CNN, each trial logged as MLflow child run
- **Golden test gate** — new model must outperform current `@production` on held-out 15% golden set before promotion
- **MLflow aliases** — `@production` and `@production_previous` used instead of deprecated stages
- **CI/CD trigger** — `model_registry_log.txt` written only when new model passes golden test → git push → GitHub Actions
- **Drift detection** — Z-score computed against baseline statistics from training data
- **APPLY_ADVANCED** — read from MLflow run tags at FastAPI startup to match model's preprocessing version
- **Streamlit metrics** — session tracking exposed on port 8002, scraped by Prometheus

### Assumptions

| Assumption | Rationale |
|-----------|-----------|
| Golden test set (15%) fixed, never used in training | Constant benchmark across all model versions |
| `@production` alias = best deployed model | Survives restarts and code changes |
| Online serving over batch | Lab instruments produce spectra one at a time |
| Z-score threshold 0.3 indicates drift | Empirically chosen, tunable via params.yaml |
| Confidence threshold 0.70 for low-confidence flag | Below 70% warrants human review |
| Self-hosted CI/CD runner | For local deployment |
| DVC local remote at ~/dvc-remote | Configurable via .dvc/config |

---

## 4. Documentation

Complete documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture diagram, component descriptions, network topology and complete MLOps lifecycle |
| [HLD.md](docs/HLD.md) | High-level design — problem statement, technology stack, design choices, Docker structure, security and scalability |
| [LLD.md](docs/LLD.md) | Low-level design — all 8 API endpoints with I/O specs, database schema, DVC stages, MLflow tracking and Prometheus metrics |
| [TEST_PLAN.md](docs/TEST_PLAN.md) | Test strategy, 16 test cases, acceptance criteria and results |
| [USER_MANUAL.md](docs/USER_MANUAL.md) | Non-technical guide — how to upload spectra, understand results, submit feedback and FAQ |

### Accessing the Web Application

Once all services are running:

1. Open **http://localhost:8501** in your browser
2. **Predict** tab → upload `.txt` spectrum file → view predicted class, confidence
3. **Pending Feedback** tab → submit correct labels for wrong predictions
4. **Pipeline** tab → view DVC DAG, MLflow runs and Airflow DAG status
5. **Help** tab → user manual