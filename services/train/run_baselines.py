import logging
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from baselines import format_results_table, log_baselines_to_mlflow, run_all_baselines
from data import read_csv_dataset, split_dataset
from process_data import process_data
from src.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

BASELINE_EXPERIMENT = "LoanPayback-Baselines"


def fetch_latest_pytorch_val_auc(client: MlflowClient, experiment_name: str) -> float | None:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.val_auc > 0",
        order_by=["start_time DESC"],
        max_results=20,
    )
    if not runs:
        return None

    best_auc = max(run.data.metrics.get("val_auc", 0.0) for run in runs)
    return round(best_auc, 4)


def main() -> None:
    config = load_config("config/config.yaml")
    dataset_config = config.dataset
    mlflow_config = config.mlflow

    df = read_csv_dataset(Path("dataset/train.csv"), dataset_config.data_length)
    trainset, valset, testset = split_dataset(
        df,
        dataset_config.target_column,
        dataset_config.test_size_1,
        dataset_config.test_size_2,
        dataset_config.stratify,
        dataset_config.random_state,
    )

    raw_valset = valset.copy()
    trainset, scaler, encoders = process_data(trainset, train=True)
    valset, _, _ = process_data(valset, scaler, encoders, train=False)

    paid_back_rate = trainset[dataset_config.target_column].mean()
    logging.info("Train paid_back rate: %.3f", paid_back_rate)

    results = run_all_baselines(
        trainset,
        valset,
        raw_valset,
        dataset_config.target_column,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", mlflow_config.tracking_uri))
    log_baselines_to_mlflow(results, mlflow, BASELINE_EXPERIMENT)

    client = MlflowClient()
    pytorch_auc = fetch_latest_pytorch_val_auc(client, mlflow_config.experiment_name)

    print("\nBaseline comparison (validation set)\n")
    print(format_results_table(results))
    if pytorch_auc is not None:
        print(f"\nLatest PyTorch NN val_auc from '{mlflow_config.experiment_name}': {pytorch_auc:.4f}")

    best = max(results, key=lambda result: result.auc)
    print(
        f"\nBest baseline here: {best.name} (AUC={best.auc:.4f}). "
        "Compare against your PyTorch production run in MLflow."
    )
    print(f"\nMLflow experiment: {BASELINE_EXPERIMENT}")


if __name__ == "__main__":
    main()
