import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.predict import predictloan
from src.schemas import validate_payload
from src.load_inference_model import load_production_model

logging.basicConfig(level=logging.INFO)

start_time = time.time()
logging.info(f"Server started at {start_time}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    response = load_production_model()
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


@app.post("/predict")
def predict(payload: validate_payload, request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    output = predictloan(request.app.state.model, payload)
    end = time.time()
    logging.info(f"Prediction took {end - start} seconds")
    return {
        "prediction": output,
        "inference_latency(ms)": int((end - start) * 1000),
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }


@app.get("/health_check")
def check(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status_code": 200,
        "message": "Model is loaded and ready to use",
        "server_uptime(ms)": int((time.time() - start_time) * 1000),
        "model_version": request.app.state.model_version,
        "model_uri": request.app.state.model_uri,
        "model_name": request.app.state.model_name,
    }
