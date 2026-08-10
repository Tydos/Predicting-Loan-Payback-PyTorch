"""
XGBoost training, ONNX export, MLflow logging, and model promotion.

Run:
    PYTHONPATH=. python services/train/train.py
"""

import logging
import math
import os
import pickle
import tempfile
from pathlib import Path

import dagshub
import mlflow
import mlflow.onnx
import numpy as np
from core.metrics import score, xy
from core.onnx_model import create_session, export_xgboost, predict_proba
from mlflow.exceptions import MlflowException
from pipeline.prepare import PreparedData, load_and_prepare_data
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def _log_preprocessing_artifacts(scaler, encoders) -> None:
    logging.info("Logging preprocessing artifacts to MLflow")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        (tmp_path / "encoders.pkl").write_bytes(pickle.dumps(encoders))
        mlflow.log_artifact(str(tmp_path / "scaler.pkl"))
        mlflow.log_artifact(str(tmp_path / "encoders.pkl"))


def _log_metrics(result) -> None:
    mlflow.log_metric("val_auc", result.auc)
    mlflow.log_metric("val_f1", result.f1)
    mlflow.log_metric("val_accuracy", result.accuracy)
    mlflow.log_metric("val_precision", result.precision)
    mlflow.log_metric("val_recall", result.recall)


def _get_champion_val_auc(client: mlflow.MlflowClient, model_name: str) -> float | None:
    """Returns the current champion's val_auc, or None if there is no champion yet."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    return client.get_run(mv.run_id).data.metrics.get("val_auc")


def _register_and_promote(onnx_model, model_name: str, n_features: int, val_auc: float) -> int:
    logging.info("Logging ONNX model artifact")
    model_info = mlflow.onnx.log_model(
        onnx_model,
        artifact_path="model",
        input_example=np.zeros((1, n_features), dtype=np.float32),
        pip_requirements=["onnxruntime", "mlflow"],
    )

    logging.info("Registering model '%s' from %s", model_name, model_info.model_uri)
    result = mlflow.register_model(model_info.model_uri, model_name)
    version = int(result.version)

    client = mlflow.MlflowClient()
    loaded = mlflow.onnx.load_model(f"models:/{model_name}/{version}")
    session = create_session(loaded)
    prob = predict_proba(session, [0.0] * n_features)

    if not math.isfinite(prob):
        logging.warning("Sanity check failed — model version %s not aliased", version)
        return version

    champion_auc = _get_champion_val_auc(client, model_name)
    if champion_auc is None or val_auc >= champion_auc:
        client.set_registered_model_alias(model_name, "champion", str(version))
        logging.info(
            "Model version %s aliased as 'champion' (val_auc=%.4f, previous champion=%s)",
            version,
            val_auc,
            f"{champion_auc:.4f}" if champion_auc is not None else "none",
        )
    else:
        logging.info(
            "Model version %s not promoted — val_auc=%.4f does not beat current champion's %.4f",
            version,
            val_auc,
            champion_auc,
        )

    return version


def run_xgboost_training(data: PreparedData) -> None:
    target = data.config.dataset.target_column
    mlflow_config = data.config.mlflow
    n_features = len(data.config.dataset.features)

    x_train, y_train = xy(data.trainset, target)
    x_val, y_val = xy(data.valset, target)

    logging.info("Fitting XGBClassifier (default hyperparameters)")
    model = XGBClassifier()
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    metrics = score("xgboost", y_val, val_probs)
    logging.info("xgboost — AUC=%.4f, F1=%.4f", metrics.auc, metrics.f1)

    onnx_model = export_xgboost(model, n_features)

    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_param("model_type", "xgboost")
        _log_metrics(metrics)
        _log_preprocessing_artifacts(data.scaler, data.encoders)
        version = _register_and_promote(
            onnx_model, mlflow_config.model_name, n_features, metrics.auc
        )

    logging.info("XGBoost run complete (run_id=%s, model_version=%s)", run.info.run_id, version)


def main() -> None:
    try:
        data = load_and_prepare_data()

        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "pjawale")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "credit-scorer")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(data.config.mlflow.experiment_name)
        logging.info("DagsHub MLflow — repo=%s/%s", repo_owner, repo_name)

        run_xgboost_training(data)
    except Exception as exc:
        logging.error("Training failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
