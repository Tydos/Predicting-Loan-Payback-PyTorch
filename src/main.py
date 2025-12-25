from src.split_data import split_dataset
from src.process_data import process_data
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.loan_dataset import loan_dataset
import yaml
from src.model import loan_predictor
from torch.optim import Adam
from torch.optim import SGD
import torch.nn as nn
import torch
from src.config_loader import load_config
from src.test_model import dummy_test
from src.read_data import read_data
from src.train import train_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("reading data and configs")
df = read_data()
config = load_config()
config1 = config.pytorch

print("splitting data")
trainset, valset, testset = split_dataset(df, config)

print("running data processing")
trainset,scaler,encoders = process_data(trainset,train=True)
valset,_,_ = process_data(valset,scaler,encoders,train=False)
testset,_,_ = process_data(testset,scaler,encoders,train=False)

print("create pytorch dataset class")
traindataset = loan_dataset(trainset)
validationdataset = loan_dataset(valset)
testingdataset = loan_dataset(testset)

print("create pytorch data loader")
BATCH_SIZE = config1.batch_size
trainerloader = DataLoader(traindataset,batch_size=BATCH_SIZE,shuffle=True)
valloader = DataLoader(validationdataset,batch_size=BATCH_SIZE,shuffle=False)
testloader = DataLoader(testingdataset,batch_size=BATCH_SIZE,shuffle=False)

print("setting model")
EPOCH = config1.epoch
LEARNING_RATE = config1.learning_rate
WEIGHT_DECAY = config1.weight_decay
MODEL_INPUT_DIM = config1.model_input_dim
model = loan_predictor(MODEL_INPUT_DIM)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)

print("model train")
import mlflow
import matplotlib.pyplot as plt
mlflow.set_experiment("LoanPayback_Experiment")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    run_id = run.info.run_id

    trained_model, loss_history, val_loss_history, test_loss_history  = train_model(
        model,
        trainerloader,
        valloader,
        testloader,
        optimizer,
        criterion,
        config.pytorch.epoch,
        device,
        mlflow
    )

    
    epochs_range = range(1, len(loss_history) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, loss_history, label='Train Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.plot(epochs_range, test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # --- Optional: log artifacts ---
    plt.savefig("loss_curves.png")
    mlflow.log_artifact("loss_curves.png")

    #Log the model execution 
    mlflow.pytorch.log_model(model, artifact_path="models")

    # Register the model in Model Registry (creates a new version automatically)
    MODEL_NAME = "LoanPayback"
    result = mlflow.register_model(
    f"runs:/{run_id}/models",MODEL_NAME
    )
    print(f"Registered {MODEL_NAME} version {result.version}")



# Load the latest registered version for testing
client = mlflow.tracking.MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME, stages=[])
latest_version_number = max(int(v.version) for v in latest_versions)
print(f"Latest version for testing: {latest_version_number}")
model_for_test = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{latest_version_number}")

# Run dummy test
if dummy_test(model_for_test):
    print("Dummy test passed! Promoting to Production...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version_number,
        stage="Production"
    )
    print(f"Model version {latest_version_number} is now in Production.")
else:
    print("Dummy test failed. Model not promoted.")