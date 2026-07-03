import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from tqdm import tqdm


class LoanDataset(Dataset):
    TARGET = "loan_paid_back"

    def __init__(self, df: pd.DataFrame):
        X = df.drop(columns=[self.TARGET]).values.astype("float32")
        Y = df[self.TARGET].values.astype("float32")
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Single pass — returns avg loss, all labels, all predicted probabilities."""
    total_loss = 0.0
    all_labels, all_probs = [], []

    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze(-1)
            total_loss += criterion(outputs, labels).item()
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def train_model(model, trainerloader, valloader, testloader, optimizer, criterion, epochs, device, mlflow):
    loss_history, val_loss_history, test_loss_history = [], [], []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for features, labels in trainerloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features).squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(trainerloader)
        loss_history.append(avg_train_loss)

        avg_val_loss, all_labels, all_probs = _evaluate(model, valloader, criterion, device)
        val_loss_history.append(avg_val_loss)

        avg_test_loss, _, _ = _evaluate(model, testloader, criterion, device)
        test_loss_history.append(avg_test_loss)

        val_preds = (all_probs >= 0.5).astype(int)
        val_auc = roc_auc_score(all_labels, all_probs)
        val_f1 = f1_score(all_labels, val_preds)
        val_accuracy = accuracy_score(all_labels, val_preds)
        val_precision = precision_score(all_labels, val_preds)
        val_recall = recall_score(all_labels, val_preds)

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)

        logging.info(
            "Epoch %d: loss=%.4f | val_loss=%.4f, AUC=%.4f, F1=%.4f, Acc=%.4f",
            epoch + 1, avg_train_loss, avg_val_loss, val_auc, val_f1, val_accuracy,
        )

    final_metrics = {
        "val_auc": val_auc,
        "val_f1": val_f1,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
    }
    return model, loss_history, val_loss_history, test_loss_history, final_metrics
