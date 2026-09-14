from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("../../../models/model.pkl")
scaler = joblib.load("../../../models/scaler.pkl")

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
    input_dict = data.model_dump()
    input_array = np.array([list(input_dict.values())])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }