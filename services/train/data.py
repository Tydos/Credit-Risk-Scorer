import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import ValidateConfig, load_config
from src.preprocessing import process_data


@dataclass
class PreparedData:
    config: ValidateConfig
    trainset: pd.DataFrame
    valset: pd.DataFrame
    testset: pd.DataFrame
    scaler: StandardScaler
    encoders: dict[str, OrdinalEncoder]


def load_and_prepare_data(config_path: str = "src/config.yaml") -> PreparedData:
    config = load_config(config_path)
    dc = config.dataset

    df = pd.read_csv("dataset/train.csv")
    logging.info("Data loaded: %s rows", len(df))

    df = df[dc.features + [dc.target_column]]
    logging.info("Feature columns selected: %s", dc.features)

    trainset, temp = train_test_split(
        df,
        test_size=dc.test_size_1,
        stratify=df[dc.target_column] if dc.stratify else None,
        random_state=dc.random_state,
    )
    valset, testset = train_test_split(
        temp,
        test_size=dc.test_size_2,
        stratify=temp[dc.target_column] if dc.stratify else None,
        random_state=dc.random_state,
    )
    logging.info("Split — train: %d, val: %d, test: %d", len(trainset), len(valset), len(testset))

    target = dc.target_column
    logging.info(
        "Class balance — paid_back rate: train=%.3f, val=%.3f, test=%.3f",
        trainset[target].mean(), valset[target].mean(), testset[target].mean(),
    )

    logging.info("Fitting scaler and encoders on training set")
    trainset, scaler, encoders = process_data(trainset, train=True)
    valset, _, _ = process_data(valset, scaler, encoders, train=False)
    testset, _, _ = process_data(testset, scaler, encoders, train=False)
    logging.info("Preprocessing complete (%d encoders fitted)", len(encoders))

    return PreparedData(
        config=config,
        trainset=trainset,
        valset=valset,
        testset=testset,
        scaler=scaler,
        encoders=encoders,
    )
