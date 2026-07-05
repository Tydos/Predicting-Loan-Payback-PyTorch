import pytest

from core.config import load_config


def test_load_config_reads_project_defaults():
    config = load_config("core/config.yaml")

    assert config.dataset.target_column == "loan_paid_back"
    assert len(config.dataset.features) == 10
    assert config.mlflow.model_name == "LoanPayback"
    assert config.inference.prediction_threshold == 0.5


def test_load_config_requires_path():
    with pytest.raises(ValueError, match="Config file path must be provided"):
        load_config(None)
