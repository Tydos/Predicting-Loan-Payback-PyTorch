from fastapi import FastAPI
from src.predict import predictloan
from src.schema import validate_payload
from src.load_inference_model import load_production_model
app = FastAPI()

@app.get("/")
def hello():
    return{"message":"API endpoint is active"}

@app.get("/predict")
def predict():
    val = 0
    return{"message":f"success {val}"}

# @app.post("/predict1")
# def predict1(request:validate_payload):
#     model, version = load_production_model()
#     if(model==-1):
#         return{"Model loading error"}
#     else:
#         output = predictloan(model,request)
#         return{"Output":f"{output}"}


# @app.get("/health_check")
# def check():
#     model, version = load_production_model()
#     return{
#         "Model Version":f"{version}"
#     }
