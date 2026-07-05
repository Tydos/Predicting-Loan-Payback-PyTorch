"""
Experimental PyTorch neural network training (not promoted to production).

Run:
    PYTHONPATH=. python services/train/pytorch_train.py
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from core.architecture import LoanPredictor
from core.config import load_config
from pipeline.prepare import load_and_prepare_data
from torch.optim import Adam
from torch.utils.data import DataLoader
from trainer import LoanDataset, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def _log_artifacts(scaler, encoders, loss_history, val_loss_history, test_loss_history) -> None:
    logging.info("Logging artifacts to MLflow (scaler, encoders, loss curve)")
    epochs_range = range(1, len(loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss_history, label="Train Loss")
    plt.plot(epochs_range, val_loss_history, label="Validation Loss")
    plt.plot(epochs_range, test_loss_history, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.grid(True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        (tmp_path / "encoders.pkl").write_bytes(pickle.dumps(encoders))
        plt.savefig(tmp_path / "loss_curves.png")
        mlflow.log_artifact(str(tmp_path / "scaler.pkl"))
        mlflow.log_artifact(str(tmp_path / "encoders.pkl"))
        mlflow.log_artifact(str(tmp_path / "loss_curves.png"))

    plt.close()


def _log_model(model, input_dim: int, model_name: str) -> int:
    logging.info("Logging PyTorch model (experimental — not promoted to champion)")
    model_info = mlflow.pytorch.log_model(
        model,
        name="models",
        input_example=torch.zeros(1, input_dim, dtype=torch.float32),
        pip_requirements=["torch", "mlflow"],
        serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
    )
    result = mlflow.register_model(model_info.model_uri, model_name)
    return int(result.version)


def main() -> None:
    try:
        logging.info("Using device: %s", device)
        data = load_and_prepare_data()
        config = load_config("core/config.yaml")
        if config.pytorch is None:
            raise ValueError("pytorch config section required for experimental training")

        mc = config.pytorch
        mlflow_config = config.mlflow

        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "pjawale")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "credit-scorer")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(mlflow_config.experiment_name)
        logging.info("DagsHub MLflow — repo=%s/%s", repo_owner, repo_name)

        trainerloader = DataLoader(
            LoanDataset(data.trainset), batch_size=mc.batch_size, shuffle=True
        )
        valloader = DataLoader(LoanDataset(data.valset), batch_size=mc.batch_size, shuffle=False)
        testloader = DataLoader(LoanDataset(data.testset), batch_size=mc.batch_size, shuffle=False)

        model = LoanPredictor(mc.model_input_dim, mc.hidden_layers, mc.dropout).to(device)
        optimizer = Adam(model.parameters(), lr=mc.learning_rate, weight_decay=mc.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        with mlflow.start_run(run_name="pytorch_nn") as run:
            mlflow.enable_system_metrics_logging()
            trained_model, loss_history, val_loss_history, test_loss_history, _ = train_model(
                model, trainerloader, valloader, testloader,
                optimizer, criterion, mc.epoch, device, mlflow,
            )
            _log_artifacts(
                data.scaler, data.encoders, loss_history, val_loss_history, test_loss_history
            )
            version = _log_model(trained_model, mc.model_input_dim, mlflow_config.model_name)

        logging.info("PyTorch run complete (run_id=%s, model_version=%s)", run.info.run_id, version)
    except Exception as exc:
        logging.error("Training failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
