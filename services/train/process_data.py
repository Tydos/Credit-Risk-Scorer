import logging

from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def process_data(dataset, scaler=None, encoders=None, train=True):
    # Drop unnessary columns
    if "id" in dataset.columns:
        dataset = dataset.drop("id", axis=1)

    num_cols = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "loan_paid_back" in num_cols:
        num_cols.remove("loan_paid_back")
    cat_cols = dataset.select_dtypes(include=["object"]).columns

    if train:
        # Scale the data with new scaler
        scaler = StandardScaler()
        dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

        # Encode categorical data with new encoder
        encoders = {}
        for col in cat_cols:
            le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            dataset[[col]] = le.fit_transform(dataset[[col]])
            encoders[col] = le

    else:
        # Scale the data with fitted scaler
        dataset[num_cols] = scaler.transform(dataset[num_cols])

        # Encode categorical data with fitted encoder
        for col in cat_cols:
            dataset[[col]] = encoders[col].transform(dataset[[col]])
    logging.info("Data processing completed.")

    return dataset, scaler, encoders
