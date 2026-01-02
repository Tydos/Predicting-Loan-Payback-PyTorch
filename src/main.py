from src.split_data import split_dataset
from src.process_data import process_data
from torch.utils.data import Dataset, DataLoader
from src.loan_dataset import loan_dataset
from src.model import loan_predictor
from torch.optim import Adam
import torch.nn as nn
import torch
from src.config_loader import load_config
from src.test_model import dummy_test
from src.read_data import read_data
from src.train import train_model
from src.mlflow_registry import register_and_promote
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("reading data and configs")
config = load_config()
config1 = config.pytorch
df = read_data(config.data_split.data_length)

print("splitting data")
config = config.data_split
target = config.target_column
test_size1=config.test_size_1
test_size2=config.test_size_2
random_state=config.random_state
is_stratify=config.stratify
trainset, valset, testset = split_dataset(df, target,test_size1,test_size2,is_stratify, random_state)

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
mlflow.set_tracking_uri("http://54.236.35.141:5000/")
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
        EPOCH,
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

    version = register_and_promote(
        model=model,
        test_fn=dummy_test,
        run_id=run_id   # pass run_id
    )


