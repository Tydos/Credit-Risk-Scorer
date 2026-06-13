import logging
from pathlib import Path
import pandas as pd
from typing import Optional


def read_csv_dataset(
    file_path: Path, data_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Read data from a CSV file and return it as a pandas DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = pd.read_csv(file_path, nrows=data_length)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {file_path}") from e

    logging.info(f"Read {data.shape[0]} rows from {file_path}")
    return data
