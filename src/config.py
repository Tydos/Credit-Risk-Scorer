import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    target_column: str
    test_size_1: float = Field(gt=0, lt=1)
    test_size_2: float = Field(gt=0, lt=1)
    stratify: bool
    random_state: int
    data_length: int | None = None


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
    model_name: str
    model_uri: str


class InferenceConfig(BaseModel):
    prediction_threshold: float


class ValidateConfig(BaseModel):
    dataset: DatasetConfig
    pytorch: PyTorchConfig
    mlflow: MLflowConfig
    inference: InferenceConfig


def load_config(path: str | None) -> ValidateConfig:
    if path is None:
        raise ValueError("Config file path must be provided.")

    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return ValidateConfig(**data)


class ValidatePayload(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=11,
        max_length=11,
        description="input strictly 11 features, and check for all float values",
        json_schema_extra={
            "example": [
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
        },
    )


class LoanApplicationPayload(BaseModel):
    annual_income: float = Field(..., gt=0, description="Borrower's annual income in USD")
    debt_to_income_ratio: float = Field(
        ..., ge=0, le=1, description="Debt-to-income ratio (0-1)"
    )
    credit_score: int = Field(..., ge=300, le=850, description="FICO-style credit score")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    interest_rate: float = Field(
        ..., ge=0, le=1, description="Annual interest rate as a decimal (e.g. 0.12)"
    )
    gender: str = Field(..., description="Applicant gender")
    marital_status: str = Field(..., description="Marital status")
    education_level: str = Field(..., description="Highest education level")
    employment_status: str = Field(..., description="Employment status")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    grade_subgrade: str = Field(..., description="Loan grade/subgrade (e.g. B3)")
