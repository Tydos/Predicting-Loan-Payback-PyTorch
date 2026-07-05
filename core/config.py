import yaml
from pydantic import BaseModel, Field


class RawDatasetConfig(BaseModel):
    raw_path: str = "dataset/accepted_2007_to_2018Q4.csv"
    output_path: str = "dataset/train.csv"
    raw_target_column: str = "loan_status"
    data_length: int | None = None


class DatasetConfig(BaseModel):
    train_path: str = "dataset/train.csv"
    target_column: str
    features: list[str]
    test_size_1: float = Field(gt=0, lt=1)
    test_size_2: float = Field(gt=0, lt=1)
    stratify: bool
    random_state: int


class PyTorchConfig(BaseModel):
    batch_size: int = Field(gt=0)
    epoch: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    model_input_dim: int = Field(gt=0)
    hidden_layers: list[int] = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)


class MLflowConfig(BaseModel):
    experiment_name: str
    model_name: str
    model_uri: str


class InferenceConfig(BaseModel):
    prediction_threshold: float


class ValidateConfig(BaseModel):
    raw_dataset: RawDatasetConfig
    dataset: DatasetConfig
    pytorch: PyTorchConfig | None = None
    mlflow: MLflowConfig
    inference: InferenceConfig


def load_config(path: str | None) -> ValidateConfig:
    if path is None:
        raise ValueError("Config file path must be provided.")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return ValidateConfig(**data)
