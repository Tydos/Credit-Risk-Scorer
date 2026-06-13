# This script orchestrates the entire training pipeline for a loan payback prediction model.
import logging
import os
from pathlib import Path
import pickle
import mlflow
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.split_data import split_dataset
from src.process_data import process_data
from src.config_loader import load_config
from src.test_model import dummy_test
from src.read_data import read_csv_dataset
from src.train import train_model
from src.mlflow_registry import register_and_promote
from src.loan_dataset import loan_dataset
from src.model import loan_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

try:
    logging.info(f"Using device: {device}")

    # Step 1: Load configuration and data
    config = load_config("config/config.yaml")
    dataset_config, model_config, mlflow_config = (
        config.dataset,
        config.pytorch,
        config.mlflow,
    )
    df = read_csv_dataset(Path("dataset/train.csv"), dataset_config.data_length)
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", mlflow_config.tracking_uri)
    )
    mlflow.set_experiment(mlflow_config.experiment_name)
    logging.info(f"Data loaded with shape: {df.shape}")

    # Step 2: Data splitting
    trainset, valset, testset = split_dataset(
        df,
        dataset_config.target_column,
        dataset_config.test_size_1,
        dataset_config.test_size_2,
        dataset_config.stratify,
        dataset_config.random_state,
    )
    logging.info(
        f"Data split into train: {trainset.shape}, val: {valset.shape}, test: {testset.shape}"
    )

    # Step 3: Data processing (scaling, encoding)
    trainset, scaler, encoders = process_data(trainset, train=True)
    valset, _, _ = process_data(valset, scaler, encoders, train=False)
    testset, _, _ = process_data(testset, scaler, encoders, train=False)
    logging.info(
        f"size of train, val, test after processing: {trainset.shape}, {valset.shape}, {testset.shape}"
    )

    # Step 4: Create PyTorch Datasets and DataLoaders
    traindataset = loan_dataset(trainset)
    validationdataset = loan_dataset(valset)
    testingdataset = loan_dataset(testset)
    trainerloader = DataLoader(
        traindataset, batch_size=model_config.batch_size, shuffle=True
    )
    valloader = DataLoader(
        validationdataset, batch_size=model_config.batch_size, shuffle=False
    )
    testloader = DataLoader(
        testingdataset, batch_size=model_config.batch_size, shuffle=False
    )
    logging.info("PyTorch Datasets and DataLoaders created")

    # Step 5: Model initialization
    model = loan_predictor(
        model_config.model_input_dim, model_config.hidden_layers, model_config.dropout
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )

    mlflow.enable_system_metrics_logging()
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        try:
            # Step 6: Train and log loss curves
            trained_model, loss_history, val_loss_history, test_loss_history = (
                train_model(
                    model,
                    trainerloader,
                    valloader,
                    testloader,
                    optimizer,
                    criterion,
                    model_config.epoch,
                    device,
                    mlflow,
                )
            )

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

            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open("encoders.pkl", "wb") as f:
                pickle.dump(encoders, f)
            plt.savefig("loss_curves.png")

            mlflow.log_artifact("scaler.pkl")
            mlflow.log_artifact("encoders.pkl")
            mlflow.log_artifact("loss_curves.png")

            # Step 7: Register and promote model
            version = register_and_promote(
                model=trained_model, test_fn=dummy_test, run_id=run_id
            )
            logging.info(f"Model registered with version: {version}")

        finally:
            if os.path.exists("scaler.pkl"):
                os.remove("scaler.pkl")
            if os.path.exists("encoders.pkl"):
                os.remove("encoders.pkl")
            if os.path.exists("loss_curves.png"):
                os.remove("loss_curves.png")

except Exception as e:
    logging.error("Training pipeline failed with error: %s", str(e), exc_info=True)
