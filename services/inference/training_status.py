import logging
from datetime import datetime, timezone

from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

TRACKED_METRICS = [
    "train_loss",
    "val_loss",
    "test_loss",
    "val_auc",
    "val_f1",
    "val_accuracy",
    "val_precision",
    "val_recall",
]


def _ms_to_iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _get_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"Experiment '{experiment_name}' not found")
    return experiment.experiment_id


def _serialize_run(run, metric_histories: dict | None = None) -> dict:
    latest_metrics = {key: run.data.metrics.get(key) for key in TRACKED_METRICS}
    latest_metrics = {key: value for key, value in latest_metrics.items() if value is not None}

    latest_epoch = None
    if metric_histories and metric_histories.get("val_auc"):
        latest_epoch = metric_histories["val_auc"][-1]["step"]
    elif "val_auc" in run.data.metrics:
        latest_epoch = 0

    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "experiment_id": run.info.experiment_id,
        "start_time": _ms_to_iso(run.info.start_time),
        "end_time": _ms_to_iso(run.info.end_time),
        "metrics": latest_metrics,
        "latest_epoch": latest_epoch,
        "artifact_uri": run.info.artifact_uri,
    }


def _metric_histories(client: MlflowClient, run_id: str) -> dict:
    histories = {}
    for metric in TRACKED_METRICS:
        try:
            history = client.get_metric_history(run_id, metric)
        except Exception:
            continue
        if history:
            histories[metric] = [
                {"step": point.step, "value": round(point.value, 6), "timestamp_ms": point.timestamp}
                for point in history
            ]
    return histories


def get_training_status(
    client: MlflowClient, experiment_name: str, model_name: str | None = None
) -> dict:
    experiment_id = _get_experiment_id(client, experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        return {
            "experiment_name": experiment_name,
            "training_in_progress": False,
            "latest_run": None,
            "metric_histories": {},
            "production_model_available": False,
            "message": "No training runs found",
        }

    latest = runs[0]
    metric_histories = _metric_histories(client, latest.info.run_id)
    serialized = _serialize_run(latest, metric_histories)

    production_model_available = False
    if model_name:
        try:
            production_model_available = bool(
                client.get_latest_versions(model_name, stages=["Production"])
            )
        except Exception as exc:
            logging.warning("Could not check Production model: %s", exc)

    return {
        "experiment_name": experiment_name,
        "training_in_progress": latest.info.status == RunStatus.to_string(RunStatus.RUNNING),
        "latest_run": serialized,
        "metric_histories": metric_histories,
        "production_model_available": production_model_available,
    }


def list_training_runs(client: MlflowClient, experiment_name: str, limit: int = 10) -> dict:
    experiment_id = _get_experiment_id(client, experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )
    return {
        "experiment_name": experiment_name,
        "runs": [_serialize_run(run) for run in runs],
    }


def get_training_run(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    metric_histories = _metric_histories(client, run_id)
    payload = _serialize_run(run, metric_histories)
    payload["metric_histories"] = metric_histories
    return payload


BASELINE_EXPERIMENT = "LoanPayback-Baselines"


def _get_latest_pytorch_model(
    client: MlflowClient,
    pytorch_experiment: str,
    model_name: str | None = None,
) -> dict | None:
    try:
        pytorch_experiment_id = _get_experiment_id(client, pytorch_experiment)
    except LookupError:
        return None

    runs = client.search_runs(
        experiment_ids=[pytorch_experiment_id],
        filter_string="attributes.status = 'FINISHED' AND metrics.val_auc > 0",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    run = runs[0]
    metrics = run.data.metrics
    label = "PyTorch NN"
    if model_name:
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                label = f"PyTorch NN (Production v{versions[0].version})"
        except Exception:
            pass

    return {
        "name": label,
        "run_id": run.info.run_id,
        "val_auc": metrics.get("val_auc"),
        "val_f1": metrics.get("val_f1"),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_precision": metrics.get("val_precision"),
        "val_recall": metrics.get("val_recall"),
        "is_pytorch": True,
    }


def list_baseline_results(
    client: MlflowClient,
    baseline_experiment: str = BASELINE_EXPERIMENT,
    pytorch_experiment: str | None = None,
    model_name: str | None = None,
) -> dict:
    experiment = client.get_experiment_by_name(baseline_experiment)
    if experiment is None:
        return {
            "experiment_name": baseline_experiment,
            "baselines": [],
            "pytorch_model": None,
            "pytorch_val_auc": None,
            "message": "No baseline experiment found. Run python run_baselines.py first.",
        }

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=50,
    )

    baselines = []
    seen_names: set[str] = set()
    for run in runs:
        name = run.data.params.get("baseline", run.info.run_name)
        if name in seen_names:
            continue
        seen_names.add(name)
        metrics = run.data.metrics
        baselines.append(
            {
                "name": name,
                "run_id": run.info.run_id,
                "val_auc": metrics.get("val_auc"),
                "val_f1": metrics.get("val_f1"),
                "val_accuracy": metrics.get("val_accuracy"),
                "val_precision": metrics.get("val_precision"),
                "val_recall": metrics.get("val_recall"),
                "notes": run.data.tags.get("notes"),
            }
        )

    baselines.sort(key=lambda row: row.get("val_auc") or 0, reverse=True)

    pytorch_model = None
    pytorch_val_auc = None
    if pytorch_experiment:
        pytorch_model = _get_latest_pytorch_model(client, pytorch_experiment, model_name)
        if pytorch_model and pytorch_model.get("val_auc") is not None:
            pytorch_val_auc = pytorch_model["val_auc"]

    return {
        "experiment_name": baseline_experiment,
        "baselines": baselines,
        "pytorch_model": pytorch_model,
        "pytorch_val_auc": pytorch_val_auc,
        "best_baseline": baselines[0] if baselines else None,
    }
