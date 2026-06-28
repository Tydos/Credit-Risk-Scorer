import argparse
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
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import load_and_prepare_data, PreparedData
from model import (
    LoanDataset,
    log_baselines_to_mlflow,
    run_all_baselines,
    train_model,
)
from src.model import loan_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def run_baseline_evaluation(data: PreparedData) -> None:
    target = data.config.dataset.target_column
    logging.info("Train paid_back rate: %.3f", data.trainset[target].mean())
    results = run_all_baselines(data.trainset, data.valset, target)
    log_baselines_to_mlflow(results, mlflow)


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


def _register_and_promote(model, run_id: str, model_name: str, input_dim: int) -> int:
    logging.info("Logging model artifact for run %s", run_id)
    model_info = mlflow.pytorch.log_model(
        model,
        name="models",
        input_example=torch.zeros(1, input_dim, dtype=torch.float32),
        pip_requirements=["torch", "mlflow"],
        serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
    )

    logging.info("Registering model '%s' from %s", model_name, model_info.model_uri)
    result = mlflow.register_model(model_info.model_uri, model_name)
    version = int(result.version)

    client = mlflow.MlflowClient()
    loaded = mlflow.pytorch.load_model(f"models:/{model_name}/{version}")
    loaded.eval()
    with torch.no_grad():
        output = loaded(torch.zeros(1, input_dim))

    if torch.isfinite(output).all().item():
        client.set_registered_model_alias(model_name, "champion", str(version))
        logging.info("Model version %s aliased as 'champion'", version)
    else:
        logging.warning("Sanity check failed — model version %s not aliased", version)

    return version


def run_pytorch_training(data: PreparedData) -> None:
    mc = data.config.pytorch
    mlflow_config = data.config.mlflow

    logging.info(
        "Building dataloaders (batch_size=%d, train=%d, val=%d, test=%d)",
        mc.batch_size, len(data.trainset), len(data.valset), len(data.testset),
    )
    trainerloader = DataLoader(LoanDataset(data.trainset), batch_size=mc.batch_size, shuffle=True)
    valloader = DataLoader(LoanDataset(data.valset), batch_size=mc.batch_size, shuffle=False)
    testloader = DataLoader(LoanDataset(data.testset), batch_size=mc.batch_size, shuffle=False)

    logging.info("Model: input_dim=%d, hidden_layers=%s, dropout=%.2f", mc.model_input_dim, mc.hidden_layers, mc.dropout)
    model = loan_predictor(mc.model_input_dim, mc.hidden_layers, mc.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=mc.learning_rate, weight_decay=mc.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    logging.info("Starting MLflow run (experiment=%s, epochs=%d, lr=%s)", mlflow_config.experiment_name, mc.epoch, mc.learning_rate)

    with mlflow.start_run(run_name="pytorch_nn") as run:
        mlflow.enable_system_metrics_logging()
        run_id = run.info.run_id
        trained_model, loss_history, val_loss_history, test_loss_history, _ = train_model(
            model, trainerloader, valloader, testloader,
            optimizer, criterion, mc.epoch, device, mlflow,
        )
        _log_artifacts(data.scaler, data.encoders, loss_history, val_loss_history, test_loss_history)
        version = _register_and_promote(trained_model, run_id, mlflow_config.model_name, mc.model_input_dim)

    logging.info("PyTorch run complete (run_id=%s, model_version=%s)", run_id, version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sklearn baselines and PyTorch training on the same split."
    )
    parser.add_argument("--baselines-only", action="store_true", help="Evaluate sklearn baselines only.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baselines; run PyTorch only.")
    args = parser.parse_args()

    if args.baselines_only and args.skip_baselines:
        raise SystemExit("Use only one of --baselines-only or --skip-baselines.")

    try:
        logging.info("Using device: %s", device)
        data = load_and_prepare_data()

        experiment_name = data.config.mlflow.experiment_name
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "pjawale")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "credit-scorer")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(experiment_name)
        logging.info("DagsHub MLflow — repo=%s/%s, experiment=%s", repo_owner, repo_name, experiment_name)

        if not args.skip_baselines:
            logging.info("Running sklearn baseline evaluation")
            run_baseline_evaluation(data)
            logging.info("Baseline evaluation complete")

        if not args.baselines_only:
            logging.info("Running PyTorch training")
            run_pytorch_training(data)
    except Exception as exc:
        logging.error("Training pipeline failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
