from pathlib import Path

import pandas as pd

from core.config import RawDatasetConfig
from pipeline.etl import (
    build_training_set,
    drop_columns,
    drop_duplicates_and_nan,
    handle_missing_values,
)


RAW_COLUMNS = [
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


def _write_raw_csv(path: Path) -> None:
    rows = [
        {
            "loan_amnt": 10_000,
            "annual_inc": 80_000,
            "dti": 15.0,
            "fico_range_low": 710,
            "int_rate": "12.5%",
            "installment": 350,
            "revol_util": "35.0%",
            "grade": "B",
            "term": " 36 months",
            "purpose": "debt_consolidation",
            "loan_status": "Fully Paid",
        },
        {
            "loan_amnt": 12_000,
            "annual_inc": 55_000,
            "dti": 22.0,
            "fico_range_low": 640,
            "int_rate": "18.0%",
            "installment": 420,
            "revol_util": "60.0%",
            "grade": "D",
            "term": "60 months",
            "purpose": "credit_card",
            "loan_status": "Charged Off",
        },
        {
            "loan_amnt": 8_000,
            "annual_inc": 45_000,
            "dti": 18.0,
            "fico_range_low": 680,
            "int_rate": "15.0%",
            "installment": 280,
            "revol_util": "40.0%",
            "grade": "C",
            "term": "36 months",
            "purpose": "medical",
            "loan_status": "Current",
        },
    ]
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)


def test_drop_columns_ignores_missing_columns():
    frame = pd.DataFrame({"a": [1], "b": [2]})

    cleaned = drop_columns(frame, ["b", "missing"])

    assert list(cleaned.columns) == ["a"]


def test_handle_missing_values_fills_numeric_and_categorical():
    frame = pd.DataFrame(
        {
            "num": [1.0, None, 3.0],
            "cat": ["x", None, "x"],
        }
    )

    filled = handle_missing_values(frame)

    assert filled["num"].isna().sum() == 0
    assert filled["cat"].isna().sum() == 0


def test_drop_duplicates_and_nan_removes_invalid_target_rows():
    frame = pd.DataFrame(
        {
            "loan_paid_back": [1, None, 1],
            "loan_amount": [1000, 2000, 1000],
        }
    )

    cleaned = drop_duplicates_and_nan(frame, target_col="loan_paid_back")

    assert len(cleaned) == 1


def test_build_training_set_filters_renames_and_writes_output(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "train.csv"
    _write_raw_csv(raw_path)

    config = RawDatasetConfig(
        raw_path=str(raw_path),
        output_path=str(output_path),
        raw_target_column="loan_status",
        data_length=None,
    )

    result = build_training_set(config)

    assert output_path.exists()
    assert "loan_paid_back" in result.columns
    assert "loan_status" not in result.columns
    assert set(result["loan_paid_back"].unique()) == {0, 1}
    assert result["interest_rate"].max() <= 1
    assert result["term"].iloc[0] == "36 months"
    assert len(result) == 2
