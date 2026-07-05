import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from core.config import ValidateConfig, load_config
from core.preprocessing import process_data


@pytest.fixture
def config() -> ValidateConfig:
    return load_config("core/config.yaml")


@pytest.fixture
def sample_training_frame(config: ValidateConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "loan_amount": 10_000.0,
                "annual_income": 75_000.0,
                "debt_to_income_ratio": 0.15,
                "credit_score": 720,
                "interest_rate": 0.12,
                "installment": 350.0,
                "revol_util": 0.35,
                "grade": "B",
                "term": "36 months",
                "loan_purpose": "debt_consolidation",
                "loan_paid_back": 1,
            },
            {
                "loan_amount": 15_000.0,
                "annual_income": 55_000.0,
                "debt_to_income_ratio": 0.28,
                "credit_score": 640,
                "interest_rate": 0.18,
                "installment": 420.0,
                "revol_util": 0.62,
                "grade": "D",
                "term": "60 months",
                "loan_purpose": "credit_card",
                "loan_paid_back": 0,
            },
        ]
    )


@pytest.fixture
def fitted_preprocessing(sample_training_frame: pd.DataFrame):
    feature_frame = sample_training_frame.drop(columns=["loan_paid_back"])
    processed, scaler, encoders = process_data(feature_frame, train=True)
    return scaler, encoders, processed.columns.tolist()


@pytest.fixture
def sample_application() -> dict:
    return {
        "loan_amount": 12_000.0,
        "annual_income": 80_000.0,
        "debt_to_income_ratio": 0.2,
        "credit_score": 710,
        "interest_rate": 0.11,
        "installment": 380.0,
        "revol_util": 0.4,
        "grade": "B",
        "term": "36 months",
        "loan_purpose": "home_improvement",
    }
