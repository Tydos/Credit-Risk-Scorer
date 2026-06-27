## Credit Risk Scorer

End-to-end MLOps for binary loan payback classification: PyTorch training, sklearn baselines, MLflow registry, FastAPI inference, Docker Compose local stack, and AWS deployment via ECR/EC2.

---

## What it does


<video src="docs/demo.mp4" controls width="720">
  Demo video
</video>

<!-- Predicts `loan_paid_back` (binary) from 11 application features. The pipeline trains a feedforward neural network, logs sklearn baselines to MLflow, registers the best run, promotes a version to **Production**, and serves predictions through a REST API and static demo UI.

**Stack:** PyTorch · FastAPI · MLflow · PostgreSQL · MinIO · Docker · GitHub Actions · AWS (EC2, ECR, S3, IAM) -->

---

## Architecture

```
Training Pipeline → MLflow Registry → FastAPI Inference API → AWS EC2 (via ECR)
     │                    │                   │
  PyTorch NN          PostgreSQL           Docker
  11 features           + S3               + IAM
```

![endpoint](docs/predict-api.png)
![mlflow](docs/mlflow.png)

---

## Dataset

| Item | Value |
|------|--------|
| Path | `dataset/train.csv` |
| Rows | **593,994** with `data_length: null` (full file) |
| Target | `loan_paid_back` (1 = paid back, 0 = default) |
| Split | 70% train · 15% validation · 15% test (stratified) |
| Counts | 415,795 / 89,099 / 89,100 |

```yaml
# config/config.yaml
dataset:
  data_length: null   # null = full dataset; integer caps rows for dev
```

---

## Model & baselines

**Production model:** fully connected PyTorch network (`src/model.py`) on 11 preprocessed features (numeric scaling + categorical encoding in `src/preprocessing.py`).

**Baselines** (sklearn, same split): majority class, credit score rule, logistic regression, random forest, hist gradient boosting — logged to MLflow experiment `LoanPayback-Baselines`.

### Validation metrics (full dataset)

| Model | Val AUC | Val F1 |
|-------|---------|--------|
| hist_gradient_boosting | 0.9189 | 0.9429 |
| random_forest | 0.9119 | 0.9117 |
| **PyTorch (Production)** | **0.9089** | 0.9413 |
| logistic_regression | 0.8921 | 0.8888 |
| credit_score_rule | 0.6055 | 0.8073 |
| majority_class | 0.5000 | 0.8882 |

| Metric | Description |
|--------|-------------|
| **AUC** | ROC area under curve on validation set; ranking quality |
| **F1** | F1 at 0.5 threshold on payback probability |
| **Threshold** | 0.5 on payback probability (`config/config.yaml` → `inference.prediction_threshold`) |

Re-run:

```bash
docker compose run --rm baselines
docker compose run --rm train
```

---

## Inference API

FastAPI app (`services/inference/main.py`) loads the Production model and preprocessing artifacts from MLflow at startup.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Static demo UI |
| `POST /predict/application` | Raw application JSON → preprocess → predict |
| `POST /predict` | 11 preprocessed floats → predict |
| `GET /training/baselines` | Baseline + PyTorch comparison JSON |
| `GET /training/status` | Latest MLflow training run |
| `GET /health_check` | Model version, preprocessing availability |
| `GET /schema` | Feature list and threshold |

Dockerfile: `services/inference/inference.dockerfile`  
p99 latency ~**450ms** under `wrk` load test (environment-dependent).

---

## Local development

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| Inference + UI | http://localhost:8000 |
| MLflow | http://localhost:5000 |

```bash
docker compose run --rm train       # train + register + promote
docker compose run --rm baselines   # sklearn baselines only
docker compose build inference && docker compose up -d inference
```

Compose mounts `./dataset` and `./config` into train/baselines containers.

---

## CI/CD

GitHub Actions (`deploy.yaml`): build image → push to ECR → SSH deploy to EC2.

---

## AWS deployment

EC2 instance with IAM role. Custom `mlflow_s3_access` policy for artifact bucket reads.

| Policy | Purpose |
|--------|---------|
| `AmazonEC2ContainerRegistryPullOnly` | Pull from ECR |
| `AmazonEC2ContainerRegistryReadOnly` | ECR metadata |
| `mlflow_s3_access` | `s3:GetObject` + `s3:ListBucket` on MLflow bucket |

---

## Project layout

```
config/config.yaml          # dataset, pytorch, mlflow, inference settings
services/train/             # training pipeline, baselines, promotion
services/inference/         # FastAPI, static UI, MLflow queries
src/                        # model, preprocessing, pydantic config
docker-compose.yml          # postgres, minio, mlflow, train, baselines, inference
```

---

## Future work

- Canary / blue-green deployments
- Auto Scaling Group + Application Load Balancer
- Drift monitoring (PSI) and outcome tracking
- Promotion gate vs baseline AUC before Production deploy
