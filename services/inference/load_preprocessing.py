import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class PreprocessingArtifacts:
    scaler: StandardScaler
    encoders: dict[str, OrdinalEncoder]


def load_preprocessing_artifacts(
    client: MlflowClient, model_name: str, version: int
) -> PreprocessingArtifacts | None:
    try:
        model_version = client.get_model_version(model_name, str(version))
        run_id = model_version.run_id
    except Exception as exc:
        logging.warning("Could not resolve MLflow run for preprocessing: %s", exc)
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            client.download_artifacts(run_id, "scaler.pkl", str(tmp_path))
            client.download_artifacts(run_id, "encoders.pkl", str(tmp_path))
        except Exception as exc:
            logging.warning("Preprocessing artifacts not found in MLflow run: %s", exc)
            return None

        with open(tmp_path / "scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(tmp_path / "encoders.pkl", "rb") as encoders_file:
            encoders = pickle.load(encoders_file)

    logging.info("Loaded preprocessing artifacts from MLflow run %s", run_id)
    return PreprocessingArtifacts(scaler=scaler, encoders=encoders)
