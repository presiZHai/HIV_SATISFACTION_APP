# ==========================
# app/model_utils.py
# ==========================
import joblib
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/catboost_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../model/encoder.joblib")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def predict(data: pd.DataFrame):
   categorical_cols = data.select_dtypes(include="object").columns.tolist()
   data[categorical_cols] = encoder.transform(data[categorical_cols])
   prediction = model.predict(data)
   prob = model.predict_proba(data)
   return prediction, prob

def get_model():
   return model

def get_encoder():
   return encoder