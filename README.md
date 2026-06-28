## Credit Risk Scorer

Credit Risk Scorer is a containerized MLOps pipeline that trains a PyTorch neural network on ~594,000 historical loan records (sourced from kaggle) to predict whether a borrower will repay or default. Training runs are tracked remotely on DagsHub's hosted MLflow, which handles
experiment logging, artifact storage (scaler, encoders, loss curves), model registration, and the champion alias that gates production promotion. The inference service is a FastAPI app that loads the champion model at startup, serves a built-in UI for manual loan
application scoring, and exposes endpoints for prediction, model validation results, and health checking — all packaged as Docker images orchestrated via Docker Compose.


## What it does


<video src="docs/demo.mp4" controls width="720">
  Demo video
</video>
---

## How to run

```python
python services/train/train.py
uvicorn services.inference.main:app --reload
```