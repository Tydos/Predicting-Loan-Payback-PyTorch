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
from src.pydantic import validate_configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load config 
print("loading configuration")
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return validate_configs(**data)


df = pd.read_csv("dataset/train.csv")
df = df.sample(10000)
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
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# --- Set experiment ---
mlflow.set_experiment("LoanPayback_Experiment")
mlflow.pytorch.autolog()

# Enable system metrics logging
mlflow.enable_system_metrics_logging()

# Start MLflow run
with mlflow.start_run():

    loss_history = []
    val_loss_history = []
    test_loss_history = []

    model.train()
    for epoch in tqdm(range(EPOCH), desc="Epochs"):
        total_loss = 0

        # --- Training loop ---
        for features, labels in trainerloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(trainerloader)
        loss_history.append(avg_train_loss)

        # --- Validation loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in valloader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)
                
                val_outputs = model(val_features).squeeze(-1)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(valloader)
        val_loss_history.append(avg_val_loss)

        # --- Test evaluation ---
        test_loss = 0
        with torch.no_grad():
            for test_features, test_labels in testloader:
                test_features = test_features.to(device)
                test_labels = test_labels.to(device)
                
                test_outputs = model(test_features).squeeze(-1)
                test_loss += criterion(test_outputs, test_labels).item()

        avg_test_loss = test_loss / len(testloader)
        test_loss_history.append(avg_test_loss)

        # --- Log metrics per epoch ---
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Test Loss={avg_test_loss:.4f}")

        model.train()  # Back to train mode

    # --- Plot loss curves ---
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
    plt.show()

    # --- Optional: log artifacts ---
    plt.savefig("loss_curves.png")
    mlflow.log_artifact("loss_curves.png")

#MLFlow logging and Promotion
MODEL_NAME = "LoanPayback"

# Dummy test function
def dummy_test(model):
    test_input = torch.zeros(1, 11)  # batch of 1, 11 features
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    return torch.isfinite(output).all().item()

# Train or initialize your model
model = loan_predictor(11)

# Start MLflow run and log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, artifact_path="models")
    run_id = run.info.run_id

# Register the model in Model Registry (creates a new version automatically)
result = mlflow.register_model(
    f"runs:/{run_id}/models",
    MODEL_NAME
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