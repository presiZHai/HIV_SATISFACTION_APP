# ==========================
# app/explanation_engine.py
# ==========================
import shap
import joblib
import pandas as pd
import os
import json
import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

model_path = os.path.join(os.path.dirname(__file__), "../model/catboost_model.joblib")
encoder_path = os.path.join(os.path.dirname(__file__), "../model/encoder.joblib")
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

label_map = {1: 'Not Satisfied', 2: 'Satisfied', 3: 'Very Satisfied'}

RULES = [
    ('Empathy was low', "Enhance provider's empathetic communication", lambda s: s.get('Empathy_Score', 3) < 2.5),
    ('Decision-sharing was low', "Improve patient engagement in decisions", lambda s: s.get('Decision_Share_Score', 3) < 2.5),
    ('Listening was moderate', "Train providers on active listening techniques", lambda s: s.get('Listening_Score', 3) < 3)
]

def enforce_categorical_dtypes(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df

def deepseek_generate_explanation(prediction, confidence, top_features, reasons, suggestions):
    prompt = f"""
You are an AI assistant helping a healthcare team understand why a specific HIV client was predicted to be '{prediction}' with {confidence}% confidence.
Top contributing factors:
{json.dumps(top_features, indent=2)}
Rule-based issues:
{reasons}
Suggestions for improvement:
{suggestions}
"""
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "deepseek/deepseek-v3-base:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "LLM error: " + response.text
    except Exception as e:
        return f"Exception: {e}"

def explain_prediction(idx, instance, model, background_data, categorical_cols):
    instance = enforce_categorical_dtypes(instance.copy(), categorical_cols)
    background_data = enforce_categorical_dtypes(background_data.copy(), categorical_cols)

    explainer = shap.TreeExplainer(model, background_data)
    shap_vals = explainer.shap_values(instance)
    preds = model.predict_proba(instance)[0]
    pred_class = model.predict(instance)[0]
    confidence = round(float(np.max(preds)) * 100, 1)
    shap_vals_row = shap_vals[np.argmax(preds)][0] if isinstance(shap_vals, list) else shap_vals[0]

    shap_dict = dict(zip(instance.columns, shap_vals_row.flatten()))
    top_features = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    top_features = {k: round(float(v), 1) for k, v in top_features.items()}

    shap_scores = {
        'Empathy_Score': instance['Empathy_Score'].iloc[0],
        'Decision_Share_Score': instance['Decision_Share_Score'].iloc[0],
        'Listening_Score': instance['Listening_Score'].iloc[0],
    }

    reasons, suggestions = [], []
    for reason_text, suggestion_text, rule_fn in RULES:
        if rule_fn(shap_scores):
            reasons.append(reason_text)
            suggestions.append(suggestion_text)

    mapped_pred = label_map.get(int(pred_class), f"Unknown class {pred_class}")

    genai_explanation = deepseek_generate_explanation(mapped_pred, confidence, top_features, reasons, suggestions)

    log_entry = {
        'instance_idx': idx,
        'prediction': mapped_pred,
        'confidence': f"{confidence}%",
        'top_features': top_features,
        'reason': "; ".join(reasons),
        'suggestions': "; ".join(suggestions),
        'genai_explanation': genai_explanation,
        'shap_values': shap_vals_row.tolist()
    }

    cache_path = os.path.join(os.path.dirname(__file__), "logs_cache.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(cache_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Failed to write to logs_cache.json: {e}")

    return log_entry