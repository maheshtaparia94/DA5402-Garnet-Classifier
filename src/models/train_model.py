import os
import pickle
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import random
import shap
import torch
import uuid

from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from mlflow.models import infer_signature

from src.utils import load_config, get_logger
from src.models.train_plsda import train as train_plsda, predict as pred_plsda
from src.models.train_svm import train as train_svm, predict as pred_svm
from src.models.train_rf import train as train_rf, predict as pred_rf
from src.models.train_mlp import train as train_mlp, predict as pred_mlp
from src.models.train_cnn import train as train_cnn, predict as pred_cnn

load_dotenv()
logger = get_logger(__name__)
config = load_config()
SEED = config["split"]["random_state"]


def get_git_commit():
    """Return current git commit hash."""
    try:
        return subprocess.getoutput("git rev-parse --short HEAD")
    except Exception:
        return "unknown"


def log_artifacts(run_dir, X_test, y_true, y_pred, class_names, model_name):
    """
    Save and log to MLflow parent run:
      - confusion_matrix.png
      - classification_report.txt
      - precision_recall_curve.png
    """
    n_classes = len(class_names)

    # test sample prediction
    wv = np.linspace(200, 1300, X_test.shape[1])
    idx = np.where(y_true == y_true[0])[0]
    sample = X_test[idx[0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wv, sample, color="steelblue")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    true_cls = class_names[int(y_true[idx[0]])]
    pred_cls = class_names[int(y_pred[idx[0]])]
    ax.set_title(
        f"Sample Spectrum | True: {true_cls} | Predicted: {pred_cls}")
    sample_path = run_dir / "sample_prediction.png"
    fig.savefig(sample_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(sample_path))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    cm_path = run_dir / "confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0)
    rp_path = run_dir / "classification_report.txt"
    rp_path.write_text(report)
    mlflow.log_artifact(str(rp_path))

    # Precision-Recall curve per class
    Y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    Y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(class_names):
        p, r, _ = precision_recall_curve(Y_bin[:, i], Y_pred_bin[:, i])
        ap = average_precision_score(Y_bin[:, i], Y_pred_bin[:, i])
        ax.plot(r, p, label=f"{cls} AP={ap:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    pr_path = run_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(pr_path))

    logger.info(f"Artifacts saved → {run_dir}")


def run_model(name, train_fn, pred_fn,
              X_train, y_train, X_val, y_val, X_test, y_test,
              n_classes, class_names, config, data_version, 
              pipeline_run_id):
    """
    Train one model type.

    Structure:
      Parent run: "{name}_experiment"
        ├── Child runs from train_fn (one per HP trial)
        └── Parent logs: best params, test metrics,
                         artifacts, model with signature

    Returns test f1_weighted for model comparison.
    """
    logger.info(f"Training {name}...")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Parent run
    with mlflow.start_run(run_name=f"{name}_experiment"):

        # Tags
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("apply_advanced",
                       str(config["preprocessing"]["apply_advanced"]))
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)

        # CV on X_train → child runs logged inside
        model, best_params, cv_score = train_fn(X_train, y_train, n_classes)

        # Refit best model on X_train + X_val
        if name in ["plsda", "svm", "rf"]:
            from sklearn.base import clone
            final_model = clone(model)
            safe_params = {
                k: v for k, v in best_params.items()
                if k in final_model.get_params()}
            final_model.set_params(**safe_params)
            final_model.fit(X_trainval, y_trainval)
            y_pred = np.array(final_model.predict(X_test)).astype(int)
        else:
            train_fn_map = {"mlp": train_mlp, "cnn": train_cnn}
            final_model, _, _ = train_fn_map[name](
                X_trainval, y_trainval, n_classes)
            y_pred = np.array(pred_fn(final_model, X_test)).astype(int)

        # Compute test metrics
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

        # Log best params
        mlflow.log_param("model_name", name)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("seed", SEED)
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # Log test metrics
        mlflow.log_metric("cv_best_f1", cv_score)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_weighted", f1_w)
        mlflow.log_metric("test_f1_macro", f1_macro)
        for cls, f1_cls in zip(class_names, f1_per):
            mlflow.log_metric(f"f1_{cls}", f1_cls)

        # Log artifacts
        run_dir = Path("artifacts_tmp") / f"{name}_experiment"
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_artifacts(run_dir, X_test,  y_test, y_pred, class_names, name)

        # Log model with infer_signature
        if hasattr(final_model, "predict_proba"):
            # SVM, RF → probabilities
            sig_out = final_model.predict_proba(X_test)
        elif name == "plsda":
            # PLSDA → normalize scores to probabilities
            scores = np.array(final_model.predict(X_test))
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            scores  = np.clip(scores, 0, None) + 1e-9
            sig_out = scores / scores.sum(axis=1, keepdims=True)
        else:
            # CNN, MLP → keep logits
            sig_out = y_pred

        X_sig = X_test.astype(np.float32)
        signature = infer_signature(X_sig, sig_out)
        if name in ["plsda", "svm", "rf"]:
            mlflow.sklearn.log_model(final_model,
                                     artifact_path="model", signature=signature)
        else:
            mlflow.pytorch.log_model(final_model,
                                     artifact_path="model", signature=signature)

    
        mlflow.log_dict({"classes": class_names}, "classes.json")
        # SHAP Explanation
        try:
            X_shap = X_test[:min(20, len(X_test))]
            wv = np.linspace(200, 1300, X_shap.shape[1])

            if hasattr(final_model, "predict_proba"):
                predict_fn = final_model.predict_proba
            elif hasattr(final_model, "eval"):
                def predict_fn(x):
                    final_model.eval()
                    with torch.no_grad():
                        return torch.softmax(
                            final_model(torch.FloatTensor(
                                x.astype("float32"))), dim=1).numpy()
            else:
                def predict_fn(x):
                    scores = np.array(final_model.predict(x))
                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)
                    scores = np.clip(scores, 0, None) + 1e-9
                    return scores / scores.sum(axis=1, keepdims=True)

            explainer = shap.KernelExplainer(predict_fn, X_shap[:5])
            shap_values = explainer.shap_values(X_shap, nsamples=50)

            # Mean absolute SHAP per feature → bar chart
            if isinstance(shap_values, list):
                # list of (n_samples, n_features) one per class
                importance = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                sv = np.abs(shap_values)
                # collapse all dims except features
                while sv.ndim > 1:
                    sv = sv.mean(axis=-1) if sv.shape[-1] < sv.shape[0] \
                         else sv.mean(axis=0)
                importance = sv

            # Top 20 most important wavenumbers
            top_idx = np.argsort(importance)[-20:][::-1]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(20), importance[top_idx], color="steelblue")
            ax.set_xticks(range(20))
            ax.set_xticklabels(
                [f"{wv[i]:.0f}" for i in top_idx], rotation=45, ha="right")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Mean |SHAP value|")
            ax.set_title(f"Top 20 Important Wavenumbers — {name}")
            plt.tight_layout()
            shap_path = run_dir / "shap_summary.png"
            fig.savefig(shap_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(shap_path))
            logger.info("SHAP logged")
        except Exception as e:
            logger.warning(f"SHAP skipped: {e}")

        
    logger.info(f"{name}: acc={acc:.4f} f1_w={f1_w:.4f}"
                f"f1_macro={f1_macro:.4f}")
    return f1_w


def main():
    splits = Path(config["data"]["splits_dir"])

    X_train = np.load(splits / "X_train.npy")
    X_val = np.load(splits / "X_val.npy")
    X_test = np.load(splits / "X_test.npy")
    y_train = np.load(splits / "y_train.npy")
    y_val = np.load(splits / "y_val.npy")
    y_test = np.load(splits / "y_test.npy")

    with open(splits / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    n_classes = len(le.classes_)
    class_names = list(le.classes_)
    data_version = "v2" if config["preprocessing"]["apply_advanced"] else "v1"

    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", str(uuid.uuid4()))
    Path("data/pipeline_run_id.txt").write_text(pipeline_run_id)
    logger.info(f"Pipeline run ID: {pipeline_run_id}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", 
                                      config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info(f"Experiment: {config['mlflow']['experiment_name']}")
    logger.info(f"Data version: {data_version} | Classes: {class_names}")

    models = [("plsda", train_plsda, pred_plsda),
        ("svm", train_svm, pred_svm),
        ("rf", train_rf, pred_rf),
        ("mlp", train_mlp, pred_mlp),
        ("cnn", train_cnn, pred_cnn)]

    results = {}
    for name, train_fn, pred_fn in models:
        results[name] = run_model(
            name, train_fn, pred_fn,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            n_classes, class_names,
            config, data_version,
            pipeline_run_id)

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE — RESULTS SUMMARY")
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:10s}  f1_weighted={f1:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()