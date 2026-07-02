import yaml
from pydantic import BaseModel, Field


class RawDatasetConfig(BaseModel):
    raw_path: str = "dataset/accepted_2007_to_2018Q4.csv"
    output_path: str = "dataset/train.csv"
    raw_target_column: str = "loan_status"
    data_length: int | None = None


class DatasetConfig(BaseModel):
    train_path: str = "dataset/train.csv"
    target_column: str
    features: list[str]
    test_size_1: float = Field(gt=0, lt=1)
    test_size_2: float = Field(gt=0, lt=1)
    stratify: bool
    random_state: int


class PyTorchConfig(BaseModel):
    batch_size: int = Field(gt=0)
    epoch: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    model_input_dim: int = Field(gt=0)
    hidden_layers: list[int] = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)


class MLflowConfig(BaseModel):
    experiment_name: str
    model_name: str
    model_uri: str


class InferenceConfig(BaseModel):
    prediction_threshold: float


class ValidateConfig(BaseModel):
    raw_dataset: RawDatasetConfig
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


class LoanApplicationPayload(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    annual_income: float = Field(..., gt=0, description="Annual income in USD (pre-tax)")
    debt_to_income_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly gross income")
    credit_score: int = Field(..., ge=300, le=850, description="FICO score (fico_range_low)")
    interest_rate: float = Field(..., ge=0, le=1, description="Annual interest rate as decimal, e.g. 0.12")
    installment: float = Field(..., gt=0, description="Monthly payment amount in USD")
    revol_util: float = Field(..., ge=0, le=1, description="Revolving credit utilisation rate, e.g. 0.55")
    grade: str = Field(..., description="LendingClub loan grade: A, B, C, D, E, F or G")
    term: str = Field(..., description="Loan term: '36 months' or '60 months'")
    loan_purpose: str = Field(..., description="Loan purpose, e.g. debt_consolidation, home_improvement")
