import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient

from load_inference_model import load_production_model
from load_preprocessing import load_preprocessing_artifacts
from make_prediction import predict
from mlflow_artifacts import artifact_response, get_run_artifacts
from training_status import (
    get_training_run,
    get_training_status,
    list_baseline_results,
    list_training_runs,
)
from src.config import LoanApplicationPayload, ValidatePayload, load_config
from src.preprocessing import FEATURE_COLUMNS, application_to_features

logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    app.state.mlflow_client = MlflowClient()
    config = load_config("config/config.yaml")
    app.state.config = config
    app.state.start_time = time.time()
    app.state.preprocessing = None
    app.state.model = None
    app.state.model_version = None
    app.state.model_uri = None
    app.state.model_name = config.mlflow.model_name

    try:
        result = load_production_model(
            app.state.mlflow_client, config.mlflow.model_name, config.mlflow.model_uri
        )
        app.state.model = result.model
        app.state.model_version = result.version
        app.state.model_uri = result.model_uri
        app.state.model_name = result.model_name
        app.state.preprocessing = load_preprocessing_artifacts(
            app.state.mlflow_client, result.model_name, result.version
        )
        logging.info("Production model v%s loaded", result.version)
    except RuntimeError as exc:
        logging.warning("Starting without Production model: %s", exc)

    logging.info("Server startup complete")
    yield

    app.state.model = None
    app.state.mlflow_client = None
    app.state.preprocessing = None


app = FastAPI(lifespan=lifespan, title="Credit Risk Inference System")

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def _build_prediction_response(output, request: Request, start: float) -> dict:
    fastapi_latency = time.perf_counter() - start
    paid_back_probability = output.confidence
    default_probability = round(1 - paid_back_probability, 4)

    return {
        "prediction": output.prediction,
        "prediction_label": "paid_back" if output.prediction == 1 else "default",
        "paid_back_probability": paid_back_probability,
        "default_probability": default_probability,
        "confidence": output.confidence,
        "fastapi_latency_sec": round(fastapi_latency, 4),
        "inference_latency_sec": output.model_latency_sec,
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }


@app.get("/", include_in_schema=False)
def ui():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "API endpoint is active. UI static files not found."}
    return FileResponse(index_path)


@app.get("/schema")
def feature_schema(request: Request):
    return {
        "features": FEATURE_COLUMNS,
        "preprocessing_available": request.app.state.preprocessing is not None,
        "prediction_threshold": request.app.state.config.inference.prediction_threshold,
        "target": "loan_paid_back",
        "prediction_values": {
            "1": "paid_back",
            "0": "default",
        },
    }


@app.post(
    "/predict", status_code=200, description="Predict using 11 preprocessed model features"
)
def predict_endpoint(payload: ValidatePayload, request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    output = predict(
        model=request.app.state.model,
        request=payload,
        prediction_threshold=request.app.state.config.inference.prediction_threshold,
    )
    logging.info(f"Prediction took {time.perf_counter() - start:.4f} seconds")
    return _build_prediction_response(output, request, start)


@app.post(
    "/predict/application",
    status_code=200,
    description="Predict from raw loan application fields",
)
def predict_application(payload: LoanApplicationPayload, request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.app.state.preprocessing is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing artifacts not loaded. Use /predict with scaled features.",
        )

    application = payload.model_dump()
    try:
        features = application_to_features(
            application,
            request.app.state.preprocessing.scaler,
            request.app.state.preprocessing.encoders,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}") from exc

    start = time.perf_counter()
    output = predict(
        model=request.app.state.model,
        request=ValidatePayload(features=features),
        prediction_threshold=request.app.state.config.inference.prediction_threshold,
    )
    logging.info(f"Application prediction took {time.perf_counter() - start:.4f} seconds")

    response = _build_prediction_response(output, request, start)
    response["application"] = application
    return response


@app.get("/training/status", description="Latest MLflow training run status and metrics")
def training_status(request: Request):
    try:
        return get_training_status(
            request.app.state.mlflow_client,
            request.app.state.config.mlflow.experiment_name,
            request.app.state.config.mlflow.model_name,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}") from exc


@app.get("/training/runs", description="Recent MLflow training runs")
def training_runs(request: Request, limit: int = 10):
    try:
        return list_training_runs(
            request.app.state.mlflow_client,
            request.app.state.config.mlflow.experiment_name,
            limit=limit,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}") from exc


@app.get("/training/runs/{run_id}/artifacts", description="List MLflow artifacts for a training run")
def training_run_artifacts(run_id: str, request: Request):
    try:
        return get_run_artifacts(request.app.state.mlflow_client, run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Artifacts not found: {exc}") from exc


@app.get(
    "/training/runs/{run_id}/artifacts/{artifact_path:path}",
    description="Download an MLflow artifact (graphs, plots, etc.)",
    include_in_schema=False,
)
def training_run_artifact_file(run_id: str, artifact_path: str, request: Request):
    try:
        return artifact_response(request.app.state.mlflow_client, run_id, artifact_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {exc}") from exc


@app.get("/training/runs/{run_id}", description="Training run details and metric history")
def training_run_detail(run_id: str, request: Request):
    try:
        return get_training_run(request.app.state.mlflow_client, run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {exc}") from exc


@app.get("/training/baselines", description="Sklearn baseline comparison results from MLflow")
def training_baselines(request: Request):
    try:
        return list_baseline_results(
            request.app.state.mlflow_client,
            pytorch_experiment=request.app.state.config.mlflow.experiment_name,
            model_name=request.app.state.config.mlflow.model_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}") from exc


@app.get(
    "/health_check",
    status_code=200,
    description="Check if the model is loaded and the server is running",
)
def check(request: Request):
    model_loaded = request.app.state.model is not None
    return {
        "message": "Model is loaded and ready to use"
        if model_loaded
        else "API is up; Production model not loaded yet",
        "model_loaded": model_loaded,
        "server_uptime(ms)": int((time.time() - request.app.state.start_time) * 1000),
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
        "preprocessing_available": request.app.state.preprocessing is not None,
    }
