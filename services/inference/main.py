import logging
import os
import pickle
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import dagshub
import mlflow.pytorch
import torch
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import LoanApplicationPayload, load_config
from src.preprocessing import FEATURE_COLUMNS, application_to_features

logging.basicConfig(level=logging.INFO)

CHAMPION_ALIAS = "champion"
STATIC_DIR = Path(__file__).parent / "static"


def _load_model(client: MlflowClient, model_name: str):
    """Loads a production model from mlflow"""
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        raise RuntimeError(f"No '{CHAMPION_ALIAS}' alias found for model '{model_name}'")

    uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    try:
        model = mlflow.pytorch.load_model(uri)
        model.eval()
        return model, int(mv.version), uri
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e


def _load_preprocessing(client: MlflowClient, model_name: str, version: int):
    """Gets the scaler and the encoder the prod model was trained on"""
    try:
        run_id = client.get_model_version(model_name, str(version)).run_id
    except Exception as exc:
        logging.warning("Could not resolve MLflow run for preprocessing: %s", exc)
        return None, None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            client.download_artifacts(run_id, "scaler.pkl", str(tmp_path))
            client.download_artifacts(run_id, "encoders.pkl", str(tmp_path))
        except Exception as exc:
            logging.warning("Preprocessing artifacts not found: %s", exc)
            return None, None

        scaler = pickle.loads((tmp_path / "scaler.pkl").read_bytes())
        encoders = pickle.loads((tmp_path / "encoders.pkl").read_bytes())

    logging.info("Loaded preprocessing artifacts from run %s", run_id)
    return scaler, encoders

def _require_ready(request: Request) -> None:
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if request.app.state.scaler is None:
        raise HTTPException(status_code=503, detail="Preprocessing artifacts not loaded")

def _require_admin(x_api_key:str = Header(...)) -> bool:
    """reload route is protected by a admin key"""
    expected = os.getenv("ADMIN","")
    if not expected or not secrets.compare_digest(x_api_key,expected):
        raise HTTPException(status_code=403,detail="Wrong key")

    return True
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_REPO_OWNER", "pjawale"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME", "credit-scorer"),
        mlflow=True,
    )

    # load the UI once on startup
    html = (STATIC_DIR / "index.html").read_text()
    css = (STATIC_DIR / "styles.css").read_text()
    js = (STATIC_DIR / "app.js").read_text()
    html = html.replace('<link rel="stylesheet" href="/assets/styles.css" />', f"<style>{css}</style>")
    html = html.replace('<script src="/assets/app.js"></script>', f"<script>{js}</script>")
    app.state.index_html = html

    config = load_config("src/config.yaml")
    app.state.config = config
    app.state.start_time = time.time()
    app.state.model = None
    app.state.model_version = None
    app.state.model_uri = None
    app.state.model_name = config.mlflow.model_name
    app.state.scaler = None
    app.state.encoders = None

    try:
        client = MlflowClient()
        model, version, uri = _load_model(client, config.mlflow.model_name)
        app.state.model = model
        app.state.model_version = version
        app.state.model_uri = uri
        app.state.scaler, app.state.encoders = _load_preprocessing(
            client, config.mlflow.model_name, version
        )
        logging.info("Production model v%s loaded", version)
    except RuntimeError as exc:
        logging.warning("Starting without production model: %s", exc)

    logging.info("Server startup complete")
    yield

    app.state.model = None
    app.state.scaler = None
    app.state.encoders = None


app = FastAPI(lifespan=lifespan, title="Credit Risk Inference API")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
      return HTMLResponse(request.app.state.index_html)

@app.get("/schema")
def schema(request: Request):
    return {
        "features": FEATURE_COLUMNS,
        "preprocessing_available": request.app.state.scaler is not None,
        "prediction_threshold": request.app.state.config.inference.prediction_threshold,
        "target": "loan_paid_back",
        "prediction_values": {"1": "paid_back", "0": "default"},
    }

@app.get("/reload",status_code=200,dependencies=[Depends(_require_admin)])
def reload_model(request:Request):
    """Hot reload a model """
    config = request.app.state.config
    try:
        client = MlflowClient()
        model, version, uri = _load_model(client,config.mlflow.model_name)
        scaler, encoders = _load_preprocessing(client, config.mlflow.model_name,version)
    except RuntimeError as re:
        raise HTTPException(status_code=503,detail=f"Could not reload due to {re}")
    
    request.app.state.model = model
    request.app.state.model_version = version
    request.app.state.model_uri = uri
    request.app.state.scaler = scaler
    request.app.state.encoders = encoders
    logging.info("Model reloaded — now serving v%s", version)
    return {"status": "reloaded", "model_version": version, "model_uri": uri}

@app.post("/predict", status_code=200)
def predict_endpoint(payload: LoanApplicationPayload, request: Request):
    _require_ready(request)
    application = payload.model_dump()

    try:
        features = application_to_features(
            application, request.app.state.scaler, request.app.state.encoders
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}") from exc

    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    threshold = request.app.state.config.inference.prediction_threshold

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = request.app.state.model(inputs).squeeze(-1)
    model_latency = time.perf_counter() - t0

    prob = round(torch.sigmoid(logits).item(), 4)
    pred = int(prob >= threshold)

    return {
        "prediction": pred,
        "prediction_label": "paid_back" if pred == 1 else "default",
        "paid_back_probability": prob,
        "default_probability": round(1 - prob, 4),
        "inference_latency_ms": round(model_latency*1000, 2),
        "model_version": request.app.state.model_version,
    }


@app.get("/health_check", status_code=200)
def health_check(request: Request):
    model_loaded = request.app.state.model is not None
    mlflow_ok = False
    try:
        MlflowClient().search_experiments(max_results=1)
        mlflow_ok = True
    except Exception:
        pass

    return {
        "model_loaded": model_loaded,
        "preprocessing_available": request.app.state.scaler is not None,
        "mlflow_reachable": mlflow_ok,
        "message": "ready" if model_loaded else "API up — production model not loaded",
        "model_version": request.app.state.model_version,
        "model_name": request.app.state.model_name,
        "server_uptime_ms": int((time.time() - request.app.state.start_time) * 1000),
    }


@app.get("/results")
def results(request: Request):
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(
            request.app.state.config.mlflow.experiment_name
        )
        if experiment is None:
            return {"baselines": [], "pytorch_model": None}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        baseline_names = {"majority_class", "logistic_regression"}
        baselines, seen = [], set()
        pytorch_model = None

        for run in runs:
            name = run.info.run_name
            if name in baseline_names and name not in seen:
                seen.add(name)
                baselines.append({"name": name, "val_auc": run.data.metrics.get("val_auc")})
            elif name == "pytorch_nn" and pytorch_model is None:
                version = request.app.state.model_version
                pytorch_model = {
                    "name": f"PyTorch NN v{version}" if version else "PyTorch NN",
                    "val_auc": run.data.metrics.get("val_auc"),
                }

        return {"baselines": baselines, "pytorch_model": pytorch_model}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
