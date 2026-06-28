## Credit Risk Scorer

Credit Risk Scorer is a containerized MLOps pipeline that trains a PyTorch neural network on ~594,000 historical loan records (sourced from kaggle) to predict whether a borrower will repay or default. Training runs are tracked remotely on DagsHub's hosted MLflow, which handles
experiment logging, artifact storage (scaler, encoders, loss curves), model registration, and the champion alias that gates production promotion. The inference service is a FastAPI app that loads the champion model at startup, serves a built-in UI for manual loan
application scoring, and exposes endpoints for prediction, model validation results, and health checking — all packaged as Docker images orchestrated via Docker Compose.


## What it does

<p align="center" width="100%">
<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/80669588/614281587-ba20f8aa-8fd9-4994-8236-60a81b0efde3.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260628T183104Z&X-Amz-Expires=300&X-Amz-Signature=3d9be4fa72eeee8cb236592b195cc65fbbf12fcf4d08d7dd11f683bb3b74b03b&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4" width="80%" controls></video>
</p>

---

## How to run

```python
python services/train/train.py
uvicorn services.inference.main:app --reload
```
