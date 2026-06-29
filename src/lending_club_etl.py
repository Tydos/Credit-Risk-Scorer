from pathlib import Path
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def read_chunks(file_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Read a CSV file in chunks.

    Args:
        file_path: Path to the CSV file to read
        nrows: Number of rows to read from the file. If None, reads entire file.

    Returns:
        pd.DataFrame: DataFrame containing the read data
    """
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    return df

def drop_columns():
    pass

def parse_dates():
    pass

def handle_missing_values():
    pass

def drop_duplicates_and_nan():
    pass

if __name__ == "__main__":
    file_path = Path("dataset") / "accepted_2007_to_2018Q4.csv"
    sample_data = read_chunks(
        file_path=file_path,
        nrows=10,
    )
    logging.info(f"Total columns: {len(sample_data.columns)}")
    
    numeric_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    object_cols = sample_data.select_dtypes(include=['object']).columns.tolist()
    logging.info(numeric_cols)
    logging.info(object_cols)
   