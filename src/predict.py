import logging
import torch
from src.schemas import validate_payload


def predictloan(model, request: validate_payload = None):
    if request is None:
        inputs = torch.tensor(
            data=[
                -1.0334,
                2.9195,
                -0.3585,
                -0.3576,
                -0.3022,
                1.0000,
                1.0000,
                0.0000,
                0.0000,
                4.0000,
                17.0000,
            ]
        )
    else:
        input_features = request.request
        inputs = torch.tensor(input_features, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).float()
    logging.info(
        f"Predicted probabilities: {probs.item()}, Predicted class: {preds.item()}"
    )

    return int(preds)
