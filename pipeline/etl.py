"""
Step 1 of the pipeline: raw LendingClub CSV → cleaned dataset/train.csv

Run:
    python pipeline/etl.py
"""

import logging
from pathlib import Path

import pandas as pd

from core.config import RawDatasetConfig, load_config

logging.basicConfig(level=logging.INFO)

RAW_FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "int_rate",
    "installment",
    "revol_util",
    "grade",
    "term",
    "purpose",
    "loan_status",
]

TARGET_MAP = {"Fully Paid": 1, "Charged Off": 0}

RENAME_MAP = {
    "loan_amnt": "loan_amount",
    "annual_inc": "annual_income",
    "dti": "debt_to_income_ratio",
    "fico_range_low": "credit_score",
    "int_rate": "interest_rate",
    "purpose": "loan_purpose",
}


def read_chunks(config: RawDatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(
        config.raw_path,
        nrows=config.data_length,
        usecols=RAW_FEATURE_COLUMNS,
        low_memory=False,
    )
    logging.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), config.raw_path)
    return df


def drop_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    return df.drop(columns=cols_to_drop, errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df


def drop_duplicates_and_nan(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])
    return df


def build_training_set(config: RawDatasetConfig) -> pd.DataFrame:
    """Full ETL pipeline: read raw CSV → clean → rename → encode target → save."""
    df = read_chunks(config)

    df = df[df["loan_status"].isin(TARGET_MAP)]
    df["loan_paid_back"] = df["loan_status"].map(TARGET_MAP)
    df = drop_columns(df, ["loan_status"])

    for pct_col in ["int_rate", "revol_util"]:
        if pct_col in df.columns:
            df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).str.strip().astype(float) / 100

    if "term" in df.columns:
        df["term"] = df["term"].str.strip()

    df = handle_missing_values(df)
    df = drop_duplicates_and_nan(df, target_col="loan_paid_back")
    df = df.rename(columns=RENAME_MAP)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %d rows to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    config = load_config(path="core/config.yaml")
    df = build_training_set(config.raw_dataset)
    logging.info("Columns: %s", df.columns.tolist())
    logging.info("Target distribution:\n%s", df["loan_paid_back"].value_counts())
