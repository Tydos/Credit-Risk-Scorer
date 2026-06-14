from dataclasses import dataclass
import mlflow.pytorch

import torch
from mlflow.exceptions import MlflowException


@dataclass
class ProductionModel:
    model: torch.nn.Module
    version: int
    model_name: str
    model_uri: str


def load_production_model(client, model_name: str, model_uri: str):
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production Models")

    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        return ProductionModel(
            model=model,
            version=int(versions[0].version),
            model_uri=model_uri,
            model_name=model_name,
        )
    except MlflowException as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e
