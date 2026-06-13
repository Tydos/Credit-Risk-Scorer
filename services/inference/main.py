import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.config_loader import load_config
from src.make_prediction import predict
from src.schemas import ValidatePayload
from src.load_inference_model import load_production_model

logging.basicConfig(level=logging.INFO)

start_time = time.time()
logging.info(f"Server started at {start_time}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    response = load_production_model()
    global config
    config = load_config("config/config.yaml")

    if response["status_code"] != 200:
        raise RuntimeError(response["message"])
    app.state.model = response["model"]
    app.state.model_version = response["version"]
    app.state.model_uri = response["model_uri"]
    app.state.model_name = response["model_name"]
    yield
    app.state.model = None
    app.state.model_version = None
    app.state.model_uri = None
    app.state.model_name = None


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
        prediction_threshold=config.inference.prediction_threshold,
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
        "server_uptime(ms)": int((time.time() - start_time) * 1000),
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }
