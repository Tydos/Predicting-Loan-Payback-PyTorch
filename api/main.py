"""
FastAPI Loan Prediction API
---------------------------

This application exposes endpoints for:

1. Health checking the API and model
2. Predicting loan outcomes using a production ML model
3. Triggering background model retraining
4. Checking the status and logs of retraining jobs

Core Components:
----------------
- FastAPI: ASGI framework for REST API
- src.load_inference_model: Loads production-ready ML model
- src.predict: Runs prediction logic
- src.schema: Request payload validation
- subprocess: Handles background retraining tasks
- BackgroundTasks: Runs long-running training jobs asynchronously
- uuid: Generates unique IDs for tracking retraining jobs

Endpoints:
----------
GET /                       -> API liveness check
POST /predict               -> Run inference on input data
GET /health_check           -> Return currently loaded model version
POST /retrain               -> Trigger background retraining
GET /retrain_status/{task_id} -> Check status and logs of retraining jobs

"""

from fastapi import FastAPI
from src.predict import predictloan
from src.schema import validate_payload
from fastapi import BackgroundTasks, HTTPException
import subprocess
import uuid

from src.load_inference_model import load_production_model
app = FastAPI()

@app.get("/")
def hello():
    return{"message":"API endpoint is active"}

@app.post("/predict")
def predict(request:validate_payload):
    model, version = load_production_model()
    if(model==-1):
        return{"Model loading error"}
    else:
        output = predictloan(model,request)
        return{"Output":f"{output}"}
    
@app.get("/health_check")
def check():
    model, version = load_production_model()
    return{
        "Model Version":f"{version}"
    }


retrain_jobs = {} 
retrain_logs = {} 
def run_training(task_id: str):
    retrain_jobs[task_id] = "running"
    retrain_logs[task_id] = [] 

    try:
        result = subprocess.run(
            ["python", "-m", "src.main"],
            capture_output=True,
            text=True,
            check=True
        )

        # Store output in logs
        retrain_logs[task_id].append(result.stdout)
        retrain_logs[task_id].append(result.stderr)
        retrain_jobs[task_id] = "success"
        retrain_logs[task_id].append("Training finished successfully.")

    except subprocess.CalledProcessError as e:
        retrain_jobs[task_id] = "failed"
        # Store error logs too
        retrain_logs[task_id].append(e.stdout)
        retrain_logs[task_id].append(e.stderr)
        retrain_logs[task_id].append(f"Training failed with return code {e.returncode}")

    except Exception as e:
        retrain_jobs[task_id] = "failed"
        retrain_logs[task_id].append(f"Exception: {str(e)}")

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    retrain_jobs[task_id] = "pending"
    retrain_logs[task_id] = []
    background_tasks.add_task(run_training, task_id)
    return {"status": "retraining started", "task_id": task_id}

@app.get("/retrain_status/{task_id}")
def retrain_status(task_id: str):
    status = retrain_jobs.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return {
        "task_id": task_id,
        "status": status,
        "logs": retrain_logs.get(task_id, [])
    }