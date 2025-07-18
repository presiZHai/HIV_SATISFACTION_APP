# ==========================
# app/explanation_engine.py
# ==========================
# This module provides functions to explain model predictions using SHAP and a GenAI model.
# It includes functions to generate explanations and apply rule-based reasoning for satisfaction scoring.
# ==========================

import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
import requests
import logging
from dotenv import load_dotenv
from pathlib import Path

# --------------------------
# Load environment variables
# --------------------------
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("SATISFACTION_APP_KEY")
if not API_KEY:
    raise EnvironmentError("❌ SATISFACTION_APP_KEY not found in .env file or environment")

# --------------------------
# Logging Setup
# --------------------------
logger = logging.getLogger("explain")
logger.setLevel(logging.INFO)

# --------------------------
# Constants
# --------------------------
label_map = {
    1: "Not Satisfied",
    2: "Satisfied",
    3: "Very Satisfied"
}

RULES = [
    ("Empathy was low", "Enhance empathetic communication", lambda s: s.get("Empathy_Score", 3) < 2.5),
    ("Decision‑sharing low", "Improve patient engagement", lambda s: s.get("Decision_Share_Score", 3) < 2.5),
    ("Listening moderate", "Train on active listening", lambda s: s.get("Listening_Score", 3) < 3),
]

# --------------------------
# Load model
# --------------------------
try:
    model = joblib.load("model/top10_model.joblib")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# --------------------------
# GenAI Explanation
# --------------------------
def deepseek_generate_explanation(pred, conf, topf, reasons, suggestions):
    prompt = json.dumps({
        "pred": pred,
        "confidence": conf,
        "top_features": topf,
        "reasons": reasons,
        "suggestions": suggestions
    }, indent=2)

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "tngtech/deepseek-r1t2-chimera:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"LLM error: {resp.text}")
            return f"LLM error: {resp.text}"
    except requests.exceptions.RequestException as e:
        logger.exception("🔌 LLM connection failed")
        return f"Exception: {e}"

# --------------------------
# Main SHAP + LLM Explanation
# --------------------------
def explain_prediction(idx, instance, model, categorical_cols=None, background_data=None):
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_vals = explainer.shap_values(instance)
        probs = model.predict_proba(instance)[0]
        pred_class = model.predict(instance)[0]
        confidence = round(float(np.max(probs)) * 100, 1)

        shap_row = shap_vals[np.argmax(probs)][0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_dict = dict(zip(instance.columns, shap_row.flatten()))
        topf = {
            k: round(v, 1)
            for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        }

        # Rule-based heuristics for satisfaction scoring 
        shap_scores = {
            score: instance.iloc[0].get(score, 3)
            for score in ["Empathy_Score", "Decision_Share_Score", "Listening_Score"]
        }

        reasons, sugg = [], []
        for reason_text, sug_text, fn in RULES:
            if fn(shap_scores):
                reasons.append(reason_text)
                sugg.append(sug_text)

        # Call GenAI
        gen = deepseek_generate_explanation(
            label_map.get(int(pred_class), "Unknown"),
            f"{confidence}%",
            topf,
            reasons,
            sugg
        )

        return {
            "instance_idx": idx,
            "prediction": label_map.get(int(pred_class), str(pred_class)),
            "confidence": f"{confidence}%",
            "top_features": topf,
            "suggestions": " ".join(sugg),
            "genai_explanation": gen,
            "shap_values": shap_row.tolist()
        }

    except Exception as e:
        logger.exception("❌ SHAP or model explanation failed")
        return {
            "instance_idx": idx,
            "prediction": "Error",
            "confidence": "0%",
            "top_features": {},
            "suggestions": "",
            "genai_explanation": f"Explanation failed: {e}",
            "shap_values": []
        }


# # ==========================
# # app/explanation_engine.py
# # ==========================
# # This module provides functions to explain model predictions using SHAP and a GenAI model.
# # It includes functions to generate explanations and apply rule-based reasoning for satisfaction scoring.
# # ==========================

# import shap
# import joblib
# import pandas as pd
# import numpy as np
# import os
# import json
# import requests
# import logging
# from dotenv import load_dotenv
# from pathlib import Path

# # --------------------------
# # Load environment variables
# # --------------------------
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)

# API_KEY = os.getenv("SATISFACTION_APP_KEY")
# if not API_KEY:
#     raise EnvironmentError("❌ SATISFACTION_APP_KEY not found in .env file or environment")

# # --------------------------
# # Logging Setup
# # --------------------------
# logger = logging.getLogger("explain")
# logger.setLevel(logging.INFO)

# # --------------------------
# # Constants
# # --------------------------
# label_map = {
#     1: "Not Satisfied",
#     2: "Satisfied",
#     3: "Very Satisfied"
# }

# RULES = [
#     ("Empathy was low", "Enhance empathetic communication", lambda s: s.get("Empathy_Score", 3) < 2.5),
#     ("Decision‑sharing low", "Improve patient engagement", lambda s: s.get("Decision_Share_Score", 3) < 2.5),
#     ("Listening moderate", "Train on active listening", lambda s: s.get("Listening_Score", 3) < 3),
# ]

# # --------------------------
# # Load model
# # --------------------------
# try:
#     model = joblib.load("model/top10_model.joblib")
# except Exception as e:
#     logger.error(f"❌ Failed to load model: {e}")
#     raise

# # --------------------------
# # GenAI Explanation
# # --------------------------
# def deepseek_generate_explanation(pred, conf, topf, reasons, suggestions):
#     prompt = json.dumps({
#         "pred": pred,
#         "confidence": conf,
#         "top_features": topf,
#         "reasons": reasons,
#         "suggestions": suggestions
#     }, indent=2)

#     try:
#         resp = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers={"Authorization": f"Bearer {API_KEY}"},
#             json={
#                 "model": "tngtech/deepseek-r1t2-chimera:free",
#                 "messages": [{"role": "user", "content": prompt}]
#             },
#             timeout=10
#         )
#         if resp.status_code == 200:
#             return resp.json()["choices"][0]["message"]["content"]
#         else:
#             logger.error(f"LLM error: {resp.text}")
#             return f"LLM error: {resp.text}"
#     except requests.exceptions.RequestException as e:
#         logger.exception("🔌 LLM connection failed")
#         return f"Exception: {e}"

# # --------------------------
# # Main SHAP + LLM Explanation
# # --------------------------
# def explain_prediction(idx, instance, model, categorical_cols=None, background_data=None):
#     try:
#         # SHAP setup
#         explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
#         shap_vals = explainer.shap_values(instance)
        
#         # Model predictions
#         probs = model.predict_proba(instance)[0]
#         pred_class = model.predict(instance)[0]
#         confidence = round(float(np.max(probs)) * 100, 1)

#         # Determine the correct SHAP row
#         if isinstance(shap_vals, list):
#             # Multi-class case: pick SHAP values for predicted class
#             class_index = np.argmax(probs)
#             shap_row = shap_vals[class_index][0]  # First instance
#         else:
#             # Binary classification or regression
#             shap_row = shap_vals[0]

#         # Match SHAP values to feature names
#         shap_dict = dict(zip(instance.columns, shap_row.flatten()))
#         topf = {
#             k: round(v, 1)
#             for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
#         }

#         # Satisfaction scoring heuristics
#         shap_scores = {
#             score: instance.iloc[0].get(score, 3)
#             for score in ["Empathy_Score", "Decision_Share_Score", "Listening_Score"]
#         }

#         reasons, sugg = [], []
#         for reason_text, sug_text, fn in RULES:
#             if fn(shap_scores):
#                 reasons.append(reason_text)
#                 sugg.append(sug_text)

#         # Generate LLM explanation
#         gen = deepseek_generate_explanation(
#             label_map.get(int(pred_class), "Unknown"),
#             f"{confidence}%",
#             topf,
#             reasons,
#             sugg
#         )

#         return {
#             "instance_idx": idx,
#             "prediction": label_map.get(int(pred_class), str(pred_class)),
#             "confidence": f"{confidence}%",
#             "top_features": topf,
#             "suggestions": " ".join(sugg),
#             "genai_explanation": gen,
#             "shap_values": shap_row.tolist()  # ✅ Serialized safely for frontend
#         }

#     except Exception as e:
#         logger.exception("❌ SHAP or model explanation failed")
#         return {
#             "instance_idx": idx,
#             "prediction": "Error",
#             "confidence": "0%",
#             "top_features": {},
#             "suggestions": "",
#             "genai_explanation": f"Explanation failed: {e}",
#             "shap_values": []
#         }