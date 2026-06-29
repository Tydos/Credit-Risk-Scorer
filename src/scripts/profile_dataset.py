"""Generate an HTML profile report for the training dataset."""

from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "train.csv"
REPORT_PATH = ROOT / "docs" / "train_profile.html"
SAMPLE_ROWS = 10_000


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_ROWS)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    profile = ProfileReport(
        df,
        title=f"Loan training data (first {SAMPLE_ROWS:,} rows)",
        explorative=True,
    )
    profile.to_file(REPORT_PATH)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
