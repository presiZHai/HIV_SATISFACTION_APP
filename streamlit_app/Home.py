
# ==========================================
# Streamlit UI (Home.py): streamlit_app/Home.py
# Purpose: Show logs from API with SHAP + LLM explanation
# ==========================================

import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import shap
from streamlit_shap import st_shap

st.set_page_config(page_title="Client Satisfaction Dashboard", layout="wide")
st.title("HIV Client Satisfaction Dashboard")
st.markdown("Enter client details for AI‑powered prediction & explanation")

@st.cache_resource
def load_artifacts():
    top_features = joblib.load("model/important_features.joblib")
    categories = joblib.load("model/categories.joblib")
    return top_features, categories

top_features, categories = load_artifacts()

if top_features is None:
    st.error("Fatal Error: Important features missing")
else:
    st.subheader("Client Info")
    with st.form("predict_form"):
        user_inputs = {}
        cols = st.columns(2)
        midpoint = len(top_features) // 2

        for idx, feature in enumerate(top_features):
            col = cols[0] if idx < midpoint else cols[1]
            label = feature.replace("_", " ").title()
            if feature in categories:
                user_inputs[feature] = col.selectbox(label, categories[feature], key=feature)
            else:
                user_inputs[feature] = col.number_input(label, value=1.0, format="%.2f", key=feature)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            resp = requests.post("http://localhost:8000/predict", json={"features": user_inputs})
            if resp.status_code == 200:
                res = resp.json()
                st.success(f"**Prediction:** {res['prediction']} (Confidence {res['confidence']})")
                st.json(res.get("top_features"))

                if res.get("suggestions"):
                    st.info(f"AI Suggestions: {res['suggestions']}")

                st.subheader("GenAI Explanation")
                st.markdown(res.get("genai_explanation", "No explanation"))

                st.subheader("SHAP Plot")
                try:
                    shap_vals = np.array(res["shap_values"])
                    explainer = shap.Explainer(lambda x: np.array([shap_vals]), top_features)
                    shap_object = explainer(user_inputs)
                    st_shap(shap.plots.waterfall(shap_object[0]), height=300)
                except Exception as e:
                    st.warning(f"SHAP plot not available: {e}")
                    st.write("Top 3 SHAP features:", res.get("top_features"))
            else:
                st.error(f"API Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        except Exception as e:
            st.error(f"Connection or API error: {e}")

    st.divider()
    st.subheader("Prediction Log")
    try:
        logs = requests.get("http://localhost:8000/logs").json()
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df[["instance_idx", "prediction", "confidence"]])
            if "suggestions" in logs_df.columns:
                st.write("AI Suggestions from logs:")
                st.write(logs_df["suggestions"])
            else:
                st.warning("No suggestions found in logs.")
        else:
            st.info("No logs yet")
    except Exception as e:
        st.warning(f"Could not load logs: {e}")


# # ==========================================
# #  Streamlit UI (Home.py):  streamlit_app/Home.py
# #  Purpose: Show logs from API with SHAP + LLM explanation
# # ==========================================
# # ==========================================
# # Streamlit UI (Home.py): streamlit_app/Home.py
# # Purpose: Show logs from API with SHAP + LLM explanation
# # ==========================================

# import streamlit as st
# import pandas as pd
# import requests
# import joblib
# import numpy as np
# import shap
# from streamlit_shap import st_shap

# # ---------------------------
# # App Config
# # ---------------------------
# st.set_page_config(page_title="Client Satisfaction Dashboard", layout="wide")
# st.title("HIV Client Satisfaction Dashboard")
# st.markdown("Enter client details for AI‑powered prediction & explanation")

# # ---------------------------
# # Load Model Artifacts
# # ---------------------------
# @st.cache_resource
# def load_artifacts():
#     top_features = joblib.load("model/important_features.joblib")
#     categories = joblib.load("model/categories.joblib")
#     return top_features, categories

# top_features, categories = load_artifacts()

# # ---------------------------
# # Form UI
# # ---------------------------
# if top_features is None:
#     st.error("Fatal Error: Important features missing")
# else:
#     st.subheader("Client Info")
#     with st.form("predict_form"):
#         user_inputs = {}
#         cols = st.columns(2)
#         midpoint = len(top_features) // 2

#         for idx, feature in enumerate(top_features):
#             col = cols[0] if idx < midpoint else cols[1]
#             label = feature.replace("_", " ").title()
#             if feature in categories:
#                 user_inputs[feature] = col.selectbox(label, categories[feature], key=feature)
#             else:
#                 user_inputs[feature] = col.number_input(label, value=1.0, format="%.2f", key=feature)

#         submitted = st.form_submit_button("Predict")

#     # ---------------------------
#     # Handle Prediction Response
#     # ---------------------------
#     if submitted:
#         try:
#             resp = requests.post("http://localhost:8000/predict", json={"features": user_inputs})
#             if resp.status_code == 200:
#                 res = resp.json()

#                 st.success(f"**Prediction:** {res['prediction']} (Confidence {res['confidence']})")
#                 st.json(res.get("top_features"))

#                 if res.get("suggestions"):
#                     st.info(f"AI Suggestions: {res['suggestions']}")

#                 st.subheader("GenAI Explanation")
#                 st.markdown(res.get("genai_explanation", "No explanation provided."))

#                 # ---------------------------
#                 # SHAP Plot
#                 # ---------------------------
#                 st.subheader("SHAP Plot")
#                 try:
#                     shap_vals = np.array(res["shap_values"])

#                     if shap_vals.size == 0:
#                         st.warning("No SHAP values returned.")
#                     else:
#                         feature_names = list(user_inputs.keys())
#                         feature_values = np.array(list(user_inputs.values()))

#                         explanation = shap.Explanation(
#                             values=shap_vals,
#                             base_values=0,  # You can adjust if needed
#                             data=feature_values,
#                             feature_names=feature_names
#                         )

#                         plot_type = st.radio("Choose SHAP plot type", ["Waterfall", "Bar"], horizontal=True)
#                         if plot_type == "Waterfall":
#                             st_shap(shap.plots.waterfall(explanation), height=300)
#                         else:
#                             st_shap(shap.plots.bar(explanation), height=300)

#                 except Exception as e:
#                     st.warning(f"SHAP plot not available: {e}")
#                     st.write("Top 3 SHAP features:", res.get("top_features"))

#             else:
#                 st.error(f"API Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")

#         except Exception as e:
#             st.error(f"Connection or API error: {e}")

#     # ---------------------------
#     # Prediction Log Viewer
#     # ---------------------------
#     st.divider()
#     st.subheader("Prediction Log")

#     try:
#         logs = requests.get("http://localhost:8000/logs").json()
#         if logs:
#             logs_df = pd.DataFrame(logs)
#             st.dataframe(logs_df[["instance_idx", "prediction", "confidence"]])

#             if "suggestions" in logs_df.columns:
#                 st.write("AI Suggestions from logs:")
#                 st.write(logs_df["suggestions"])
#             else:
#                 st.warning("No suggestions found in logs.")
#         else:
#             st.info("No logs yet.")
#     except Exception as e:
#         st.warning(f"Could not load logs: {e}")