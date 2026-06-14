import time
import torch
import logging
from src.config import ValidatePayload
from dataclasses import dataclass


@dataclass
class PredictionOutput:
    prediction: int
    confidence: float
    model_latency_sec: float


def predict(
    model: torch.nn.Module,
    request: ValidatePayload,
    prediction_threshold: float = 0.5,
) -> PredictionOutput:
    """
    Make a prediction using the provided model and input data.

    Args:
        model: The trained PyTorch model.
        request: The input data for prediction.
        prediction_threshold: The threshold for binary classification.

    Returns:
        PredictionOutput containing predicted class, confidence score, and model latency.
    """

    try:
        input_features = request.features
        inputs = torch.tensor(input_features, dtype=torch.float32)
    except (ValueError, TypeError, RuntimeError) as e:
        raise ValueError(f"Invalid input features: {e}") from e

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(inputs)
    model_latency = time.perf_counter() - start

    probs = torch.sigmoid(outputs)
    preds = (probs >= prediction_threshold).float()
    logging.info(
        f"Predicted probabilities: {probs.item()}, Predicted class: {preds.item()}"
    )

    return PredictionOutput(
        prediction=int(preds.item()),
        confidence=round(probs.item(), 4),
        model_latency_sec=round(model_latency, 2),
    )
