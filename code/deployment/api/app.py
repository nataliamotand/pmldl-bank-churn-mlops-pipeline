from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None
model_last_modified = None

def load_model_if_updated():
    global model, scaler, model_last_modified
    current_mtime = os.path.getmtime(MODEL_PATH)

    if model_last_modified != current_mtime:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_last_modified = current_mtime
        print("Model reloaded from disk.")

load_model_if_updated()

app = FastAPI()

class ClientData(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int
    Gender_Male: int

@app.post("/predict")
def predict(data: ClientData):
    load_model_if_updated()

    input_dict = data.model_dump()
    input_array = np.array([list(input_dict.values())])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }