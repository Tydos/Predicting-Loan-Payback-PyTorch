from pydantic import BaseModel, Field

class DataSplitConfig(BaseModel):
    target_column: str
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

class validate_configs(BaseModel):
    data_split: DataSplitConfig
    pytorch: PyTorchConfig