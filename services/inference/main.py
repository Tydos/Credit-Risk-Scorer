import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.config import load_config, ValidatePayload
from make_prediction import predict
from load_inference_model import load_production_model
from mlflow.tracking import MlflowClient

import os
import mlflow


logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    app.state.mlflow_client = MlflowClient()
    config = load_config("config/config.yaml")
    app.state.config = config
    app.state.start_time = time.time()

    try:
        result = load_production_model(
            app.state.mlflow_client, config.mlflow.model_name, config.mlflow.model_uri
        )
    except RuntimeError as e:
        raise RuntimeError(f"Startup failed: {e}") from e

    app.state.model = result.model
    app.state.model_version = result.version
    app.state.model_uri = result.model_uri
    app.state.model_name = result.model_name

    logging.info("Server startup complete")
    yield

    app.state.model = None
    app.state.mlflow_client = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
def hello():
    return {"message": "API endpoint is active"}


@app.post(
    "/predict", status_code=200, description="Predict the class based on input features"
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
    fastapi_latency = time.perf_counter() - start

    logging.info(f"Prediction took {fastapi_latency:.4f} seconds")

    return {
        "prediction": output.prediction,
        "confidence": output.confidence,
        "fastapi_latency_sec": round(fastapi_latency, 4),
        "inference_latency_sec": output.model_latency_sec,
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }


@app.get(
    "/health_check",
    status_code=200,
    description="Check if the model is loaded and the server is running",
)
def check(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "message": "Model is loaded and ready to use",
        "server_uptime(ms)": int((time.time() - request.app.state.start_time) * 1000),
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }
