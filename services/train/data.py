import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_dataset(
    file_path: Path, data_length: Optional[int] = None
) -> pd.DataFrame:
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


def split_dataset(dataset, target, test_size1, test_size2, is_stratify, random_state):
    trainset, tempset = train_test_split(
        dataset,
        test_size=test_size1,
        stratify=dataset[target] if is_stratify else None,
        random_state=random_state,
    )

    valset, testset = train_test_split(
        tempset,
        test_size=test_size2,
        stratify=tempset[target] if is_stratify else None,
        random_state=random_state,
    )

    logging.info("Dataset split completed.")
    logging.info(f"Original dataset size: {len(dataset)}")
    logging.info(f"Train set size: {len(trainset)}")
    logging.info(f"Validation set size: {len(valset)}")
    logging.info(f"Test set size: {len(testset)}")
    return trainset, valset, testset
