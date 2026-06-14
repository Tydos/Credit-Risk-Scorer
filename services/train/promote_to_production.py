import logging
import torch
import mlflow.pytorch


def dummy_test(model):
    test_input = torch.zeros(1, 11)
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    return torch.isfinite(output).all().item()


def register_and_promote(model, run_id, test_fn):
    MODEL_NAME = "LoanPayback"
    mlflow.pytorch.log_model(model, artifact_path="models")
    result = mlflow.register_model(f"runs:/{run_id}/models", MODEL_NAME)
    logging.info(f"Registered {MODEL_NAME} version {result.version}")

    client = mlflow.MlflowClient()
    latest_version = max(int(v.version) for v in client.get_latest_versions(MODEL_NAME))
    logging.info(f"Latest version for testing: {latest_version}")

    model_for_test = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{latest_version}")

    if test_fn(model_for_test):
        logging.info("Dummy test passed! Promoting to Production...")
        client.transition_model_version_stage(
            MODEL_NAME, latest_version, stage="Production"
        )
        logging.info(f"Model version {latest_version} is now in Production.")
    else:
        logging.info("Dummy test failed. Model not promoted.")

    return latest_version
