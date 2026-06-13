from src.schemas import ValidateConfig
import yaml


def load_config(path: str | None) -> ValidateConfig:
    if path is None:
        raise ValueError("Config file path must be provided.")

    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return ValidateConfig(**data)
