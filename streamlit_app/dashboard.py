# ==========================================
#  Streamlit UI (streamlit_app/dashboard.py)
# ==========================================
import streamlit as st
import pandas as pd
import requests
import shap
import numpy as np
from streamlit_shap import st_shap

st.set_page_config(page_title="Client Satisfaction Dashboard", layout="wide")
st.title("📊 HIV Client Satisfaction Dashboard")

try:
    logs = requests.get("http://localhost:8000/logs").json()
    df = pd.DataFrame(logs)

    st.dataframe(df[["instance_idx", "prediction", "confidence", "top_features"]])

    selected_idx = st.selectbox("Select an instance to view:", df['instance_idx'])
    row = df[df['instance_idx'] == selected_idx].iloc[0]

    st.subheader("Prediction Details")
    st.write("**Prediction:**", row['prediction'])
    st.write("**Confidence:**", row['confidence'])
    st.write("**Top Features:**", row['top_features'])
    st.write("**Reason:**", row['reason'])
    st.write("**Suggestions:**", row['suggestions'])

    st.subheader("SHAP Visualization")
    try:
        shap_values = np.array(row['shap_values']).reshape(1, -1)
        features = pd.DataFrame([row['top_features']])
        explainer = shap.Explanation(values=shap_values[0], data=features.values[0], feature_names=list(features.columns))
        st_shap(shap.plots.waterfall(explainer), height=400)
    except Exception as e:
        st.warning(f"SHAP plot unavailable for this entry: {e}")

    st.subheader("GenAI Explanation")
    st.markdown(row['genai_explanation'])
except Exception as e:
    st.error(f"Failed to load logs: {e}")