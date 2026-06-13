# MLflow Tracking Server Image
# ----------------------------
# This image runs an MLflow tracking server in standalone mode.
#
# Backend Store:
#   - Stores experiment metadata (e.g., PostgreSQL)
# Artifact Store:
#   - Stores model artifacts (e.g., S3)

FROM python:3.11-slim

WORKDIR /app

COPY services/mlflow-server/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts '*'