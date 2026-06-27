import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "split": self.split,
            "auc": round(self.auc, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "notes": self.notes,
        }


def _xy_frames(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    features = frame.drop(columns=[target]).values.astype("float32")
    labels = frame[target].values.astype("float32")
    return features, labels


def _score_metrics(name: str, split: str, y_true: np.ndarray, y_prob: np.ndarray, notes: str = "") -> BaselineResult:
    y_pred = (y_prob >= 0.5).astype(int)
    return BaselineResult(
        name=name,
        split=split,
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        notes=notes,
    )


def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> BaselineResult:
    positive_rate = float(np.mean(y_train))
    probabilities = np.full(shape=y_eval.shape, fill_value=positive_rate, dtype=float)
    return _score_metrics(
        "majority_class",
        "validation",
        y_eval,
        probabilities,
        notes=f"Always predicts paid_back={positive_rate:.3f}",
    )


def credit_score_rule_baseline(
    raw_eval: pd.DataFrame,
    y_eval: np.ndarray,
    threshold: int = 650,
) -> BaselineResult:
    probabilities = (raw_eval["credit_score"].values >= threshold).astype(float)
    return _score_metrics(
        "credit_score_rule",
        "validation",
        y_eval,
        probabilities,
        notes=f"Approve if credit_score >= {threshold}",
    )


def logistic_regression_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> BaselineResult:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    return _score_metrics("logistic_regression", "validation", y_eval, probabilities)


def random_forest_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> BaselineResult:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=11,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    return _score_metrics("random_forest", "validation", y_eval, probabilities)


def hist_gradient_boosting_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> BaselineResult:
    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=200,
        random_state=11,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    return _score_metrics("hist_gradient_boosting", "validation", y_eval, probabilities)


def run_all_baselines(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    raw_val_frame: pd.DataFrame,
    target: str,
) -> list[BaselineResult]:
    x_train, y_train = _xy_frames(train_frame, target)
    x_val, y_val = _xy_frames(val_frame, target)

    return [
        majority_class_baseline(y_train, y_val),
        credit_score_rule_baseline(raw_val_frame, y_val),
        logistic_regression_baseline(x_train, y_train, x_val, y_val),
        random_forest_baseline(x_train, y_train, x_val, y_val),
        hist_gradient_boosting_baseline(x_train, y_train, x_val, y_val),
    ]


def format_results_table(results: list[BaselineResult]) -> str:
    headers = ["Model", "AUC", "F1", "Accuracy", "Precision", "Recall", "Notes"]
    rows = [
        [
            result.name,
            f"{result.auc:.4f}",
            f"{result.f1:.4f}",
            f"{result.accuracy:.4f}",
            f"{result.precision:.4f}",
            f"{result.recall:.4f}",
            result.notes,
        ]
        for result in results
    ]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    lines = []
    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    lines.extend([header_line, divider])
    for row in rows:
        lines.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def log_baselines_to_mlflow(results: list[BaselineResult], mlflow_module, experiment_name: str) -> None:
    mlflow_module.set_experiment(experiment_name)
    for result in results:
        with mlflow_module.start_run(run_name=result.name):
            mlflow_module.log_param("baseline", result.name)
            mlflow_module.log_param("split", result.split)
            mlflow_module.log_metric("val_auc", result.auc)
            mlflow_module.log_metric("val_f1", result.f1)
            mlflow_module.log_metric("val_accuracy", result.accuracy)
            mlflow_module.log_metric("val_precision", result.precision)
            mlflow_module.log_metric("val_recall", result.recall)
            if result.notes:
                mlflow_module.set_tag("notes", result.notes)
    logging.info("Logged %s baseline runs to MLflow experiment '%s'", len(results), experiment_name)
