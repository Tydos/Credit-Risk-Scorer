from src.schemas import validate_configs
import yaml


def load_config(path: str | None) -> validate_configs:
    if path is None:
        raise ValueError("Config file path must be provided.")

    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return validate_configs(**data)
