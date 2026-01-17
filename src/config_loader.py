from src.schema import validate_configs
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return validate_configs(**data)
