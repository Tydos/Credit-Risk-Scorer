# This module defines the data schemas for configuration and request/response using Pydantic.
from pydantic import BaseModel, Field
from typing import Annotated


# Configuration schemas for dataset and models
class DatasetConfig(BaseModel):
    target_column: str
    test_size_1: float = Field(gt=0, lt=1)
    test_size_2: float = Field(gt=0, lt=1)
    stratify: bool
    random_state: int
    data_length: int


class PyTorchConfig(BaseModel):
    batch_size: int = Field(gt=0)
    epoch: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    model_input_dim: int = Field(gt=0)
    hidden_layers: list[int] = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class InferenceConfig(BaseModel):
    prediction_threshold: float


class ValidateConfig(BaseModel):
    dataset: DatasetConfig
    pytorch: PyTorchConfig
    mlflow: MLflowConfig
    inference: InferenceConfig


# Schema for API request payload validation
class ValidatePayload(BaseModel):
    features: list[float] = Field(
        ...,
        min_items=11,
        max_items=11,
        description="input strictly 11 features, and check for all float values",
        example=[
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
        ],
    )
