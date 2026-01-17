from src.model import loan_predictor
import mlflow.pytorch
from mlflow.tracking import MlflowClient
MODEL_NAME = "LoanPayback"

def load_production_model():
    mlflow.set_tracking_uri("http://54.236.35.141:5000/")
    model_uri = "models:/LoanPayback/Production"  # MLflow model registry stage
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"] )
    
    if(model):
        return model, int(versions[0].version)
    else:   
        return -1