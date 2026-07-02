import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def process_data(dataset, scaler=None, encoders=None, train=True):
    if "id" in dataset.columns:
        dataset = dataset.drop("id", axis=1)

    num_cols = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "loan_paid_back" in num_cols:
        num_cols.remove("loan_paid_back")
    cat_cols = dataset.select_dtypes(include=["object"]).columns

    if train:
        scaler = StandardScaler()
        dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

        encoders = {}
        for col in cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            dataset[[col]] = encoder.fit_transform(dataset[[col]])
            encoders[col] = encoder
    else:
        dataset[num_cols] = scaler.transform(dataset[num_cols])
        for col in cat_cols:
            dataset[[col]] = encoders[col].transform(dataset[[col]])

    return dataset, scaler, encoders


def application_to_features(
    application: dict,
    scaler: StandardScaler,
    encoders: dict[str, OrdinalEncoder],
    feature_columns: list[str],
) -> list[float]:
    row = pd.DataFrame([application], columns=feature_columns)
    processed, _, _ = process_data(row, scaler=scaler, encoders=encoders, train=False)
    return processed.astype("float32").values.flatten().tolist()
