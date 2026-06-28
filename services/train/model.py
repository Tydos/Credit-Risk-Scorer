import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    name: str
    split: str
    auc: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    notes: str = ""


def _xy(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.drop(columns=[target]).values.astype("float32"),
        frame[target].values.astype("float32"),
    )


def _score(name: str, y_true: np.ndarray, y_prob: np.ndarray, notes: str = "") -> BaselineResult:
    y_pred = (y_prob >= 0.5).astype(int)
    return BaselineResult(
        name=name,
        split="validation",
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        notes=notes,
    )


def run_all_baselines(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    target: str,
) -> list[BaselineResult]:
    x_train, y_train = _xy(train_frame, target)
    x_val, y_val = _xy(val_frame, target)

    logging.info("Fitting majority-class baseline")
    majority_prob = np.full(y_val.shape, float(np.mean(y_train)))
    majority = _score(
        "majority_class", y_val, majority_prob,
        f"Always predicts paid_back={majority_prob[0]:.3f}",
    )
    logging.info("majority_class — AUC=%.4f, F1=%.4f", majority.auc, majority.f1)

    logging.info("Fitting LogisticRegression baseline")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=11)
    lr.fit(x_train, y_train)
    lr_result = _score("logistic_regression", y_val, lr.predict_proba(x_val)[:, 1])
    logging.info("logistic_regression — AUC=%.4f, F1=%.4f", lr_result.auc, lr_result.f1)

    return [majority, lr_result]


def log_baselines_to_mlflow(results: list[BaselineResult], mlflow_module) -> None:
    for result in results:
        logging.info("Logging baseline '%s' to MLflow", result.name)
        with mlflow_module.start_run(run_name=result.name):
            mlflow_module.log_param("model_type", "baseline")
            mlflow_module.log_param("baseline", result.name)
            mlflow_module.log_param("split", result.split)
            mlflow_module.log_metric("val_auc", result.auc)
            mlflow_module.log_metric("val_f1", result.f1)
            mlflow_module.log_metric("val_accuracy", result.accuracy)
            mlflow_module.log_metric("val_precision", result.precision)
            mlflow_module.log_metric("val_recall", result.recall)
            if result.notes:
                mlflow_module.set_tag("notes", result.notes)
    logging.info("Logged %d baseline runs to MLflow", len(results))


# ---------------------------------------------------------------------------
# PyTorch dataset + training loop
# ---------------------------------------------------------------------------


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
