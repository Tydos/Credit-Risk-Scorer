import logging
import mimetypes
import tempfile
from pathlib import Path

from fastapi.responses import Response
from mlflow.tracking import MlflowClient

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def _list_artifact_files(client: MlflowClient, run_id: str, path: str = "") -> list[dict]:
    files = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.extend(_list_artifact_files(client, run_id, item.path))
            continue

        suffix = Path(item.path).suffix.lower()
        files.append(
            {
                "path": item.path,
                "file_size": item.file_size,
                "is_image": suffix in IMAGE_EXTENSIONS,
            }
        )
    return files


def get_run_artifacts(client: MlflowClient, run_id: str) -> dict:
    artifacts = _list_artifact_files(client, run_id)
    for artifact in artifacts:
        artifact["url"] = f"/training/runs/{run_id}/artifacts/{artifact['path']}"

    return {
        "run_id": run_id,
        "artifacts": artifacts,
        "images": [artifact for artifact in artifacts if artifact["is_image"]],
    }


def load_artifact_bytes(client: MlflowClient, run_id: str, artifact_path: str) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        file_path = Path(local_path)
        if file_path.is_dir():
            raise FileNotFoundError(f"Artifact path is a directory: {artifact_path}")

        content = file_path.read_bytes()
        media_type = mimetypes.guess_type(artifact_path)[0] or "application/octet-stream"
        return content, media_type


def artifact_response(client: MlflowClient, run_id: str, artifact_path: str) -> Response:
    try:
        content, media_type = load_artifact_bytes(client, run_id, artifact_path)
    except Exception as exc:
        logging.warning("Failed to load artifact %s for run %s: %s", artifact_path, run_id, exc)
        raise

    return Response(content=content, media_type=media_type)
