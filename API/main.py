from fastapi import FastAPI
from src.predict import predictloan
app = FastAPI()

@app.get("/")
def hello():
    return{"message":"API endpoint active"}

@app.get("/predict")
def predict():
    val = predictloan()
    return{"message":f"success {val}"}