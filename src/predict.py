import torch
import torch.nn as nn
from src.model import loan_predictor
import mlflow.pytorch

def predictloan():
    inputs = torch.tensor(data=[-1.0334,  2.9195, -0.3585, -0.3576, -0.3022,  1.0000,  1.0000,  0.0000,
            0.0000,  4.0000, 17.0000])

    model = loan_predictor(11)

    #Switch to a production model using MLFLow instead of hardcoding 
    # model.load_state_dict(torch.load("Models\V1.pt"))
    # Load the production model from MLflow
    model_uri = "models:/LoanPayback/Production"  # MLflow model registry stage
    model = mlflow.pytorch.load_model(model_uri)
    
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    # print(outputs)
    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).float()
    # print(int(preds))

    return int(preds)

print(predictloan())