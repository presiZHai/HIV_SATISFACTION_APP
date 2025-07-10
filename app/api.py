from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
from app.model_utils import model, encoder
from app.explanation_engine import explain_prediction
import os

app = FastAPI()

class Instance(BaseModel):
    features: dict

@app.post("/predict")
def predict_instance(instance: Instance):
    try:
        df = pd.DataFrame([instance.features])
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        background_data = df.sample(n=1, random_state=42)
        explanation = explain_prediction(0, df, model, background_data, categorical_cols)

        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
def get_logs():
    try:
        with open("app/logs_cache.json", "r") as f:
            logs = json.load(f)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))