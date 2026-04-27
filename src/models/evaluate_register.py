import mlflow
import os
import numpy as np
import pickle
import datetime

from dotenv import load_dotenv
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from src.utils import load_config, get_logger
from src.data.preprocessor import preprocess_spectrum


load_dotenv()
logger = get_logger(__name__)
GOLDEN_DIR = Path("data/golden")

def score_on_golden(run_id, config):
    """Load model from MLflow run, evaluate on golden test set, return f1."""
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    with open("data/splits/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    adv = run.data.tags.get("apply_advanced", "false")
    apply_advanced = adv.lower() == "true"
    y_true, y_pred = [], []
    for class_dir in sorted(GOLDEN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for fpath in class_dir.glob("*.txt"):
            wn, intensity = [], []
            for line in fpath.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        wn.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue
            if not wn:
                continue
            X = preprocess_spectrum(
                np.array(wn), np.array(intensity), config, apply_advanced).reshape(1, -1).astype(np.float32)
            result = np.array(model.predict(X)).flatten()
            # All models return probs/logits → argmax gives class index
            if result.ndim == 0 or len(result) == 1:
                pred = int(result.flat[0])  # fallback for single value
            else:
                pred = int(np.argmax(result))
            y_true.append(class_dir.name)
            y_pred.append(le.inverse_transform([pred])[0])
    if not y_true:
        return 0.0
    return round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)


def get_run(client, experiment_name):
    """Get the training run for this pipeline. Falls back to most recent."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Get pipeline_run_id
    pipeline_run_id = Path("data/pipeline_run_id.txt").read_text().strip() \
        if Path("data/pipeline_run_id.txt").exists() else \
        os.environ.get("PIPELINE_RUN_ID", "")

    filter_str = f"tags.pipeline_run_id = '{pipeline_run_id}'" \
        if pipeline_run_id else ""

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"])

    # Parent runs only
    parent_runs = [r for r in runs
                   if "mlflow.parentRunId" not in r.data.tags]

    if not parent_runs:
        raise ValueError("No runs found. Run train_model.py first.")

    run = parent_runs[0]
    logger.info(f"Run: {run.info.run_name} "
                f"f1={run.data.metrics.get('test_f1_weighted', 0):.4f}")
    return run

def register_and_deploy(client, best_run, registry_name, threshold, config):
    # Skip if same run already deployed
    try:
        current = client.get_model_version_by_alias(
            registry_name, "production")
        if current.run_id == best_run.info.run_id:
            logger.info("Best run already @production — skipping")
            return True, 0.0
    except Exception:
        current = None

    # Score new model on golden set
    logger.info("Running golden test on best model...")
    new_f1 = score_on_golden(best_run.info.run_id, config)
    logger.info(f"New model golden f1={new_f1:.4f}")

    # Compare vs current champion OR vs threshold
    if current is not None:
        current_f1 = score_on_golden(current.run_id, config)
        logger.info(f"Current @production golden f1={current_f1:.4f}")
        passes = new_f1 > current_f1
        logger.info(f"{'PASS' if passes else 'FAIL'} "
                    f"new={new_f1:.4f} vs current={current_f1:.4f}")
    else:
        passes = new_f1 >= threshold
        logger.info(f"No @production — threshold check: "
                    f"{'PASS' if passes else 'FAIL'} "
                    f"f1={new_f1:.4f} threshold={threshold}")

    if passes:
        mv = mlflow.register_model(
            f"runs:/{best_run.info.run_id}/model", registry_name)
        
        try:
            current = client.get_model_version_by_alias(
                registry_name, "production")
            client.set_registered_model_alias(
                registry_name, "production_previous", current.version)
            logger.info(f"v{current.version} → @production_previous")
        except Exception:
            pass
        # Set new production directly
        client.set_registered_model_alias(
            registry_name, "production", mv.version)
        logger.info(f"v{mv.version} → @production")
        
        # Write deployment log for CI/CD trigger
        log_path = Path("data/model_registry_log.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"version={mv.version}\n"
            f"run_id={best_run.info.run_id}\n"
            f"golden_f1={new_f1}\n"
            f"timestamp={datetime.datetime.utcnow().isoformat()}\n"
            f"pipeline_run_id={best_run.data.tags.get('pipeline_run_id', '')}\n")
        logger.info("data/model_register_log.txt updated")
    else:
        logger.info(f"f1={new_f1:.4f} below threshold → @staging NOT set")

    return passes, new_f1


def main():
    config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI",
                             config["mlflow"]["tracking_uri"])
    exp_name = config["mlflow"]["experiment_name"]
    reg_name = config["mlflow"]["registry_name"]
    threshold = config["evaluation"]["min_f1_threshold"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    #Get best run
    best_run = get_run(client, exp_name)

    #Register, validate, deploy
    passed, f1 = register_and_deploy(client, best_run, reg_name, threshold, config)

    # Summary
    logger.info("=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"  Best model : {best_run.info.run_name}")
    logger.info(f"  Staging f1 : {f1:.4f}")
    logger.info(f"  Result : {'Production' if passed else 'Pass'}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()