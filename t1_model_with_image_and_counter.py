
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- CONFIG -------------------
MODEL_DIR = "."
COUNTER_FILE = "usage_counter.txt"
IMAGE_PATH = "header_image.png"

# ------------------- PAGE SETUP -------------------
st.set_page_config(layout="wide")
st.title("Prenatal Risk Prediction Calculator")

# Display image if available
if os.path.exists(IMAGE_PATH):
    image = Image.open(IMAGE_PATH)
    st.image(image, use_column_width=True)

# ------------------- USAGE COUNTER -------------------
if not os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE, "w") as f:
        f.write("0")

with open(COUNTER_FILE, "r+") as f:
    count = int(f.read()) + 1
    f.seek(0)
    f.write(str(count))

# Display usage stats
st.markdown(
    f"""
    > 📊 *As of **15.07.2025**, predictions were made in **{count}** cases, and our model is trained on data from **4,349** patients.*
    """,
    unsafe_allow_html=True
)

# ------------------- USER INPUT COLUMNS -------------------
col1, col2 = st.columns(2)

# ------------------- PANEL 1: First Trimester Calculator -------------------
with col1:
    st.subheader("First Trimester Calculator")
    st.caption("Trained on 251 cases with cystic hygroma (CH)")

    target_1st = st.selectbox("Select 1st trimester outcome:", [
        'Composite', 'Anomaly', 'ChromsProb', 'SAB', 'Livebirth',
        "T21", "T18", "T13", "Turners", "OtherChroms"], key="1st_trimester_selector")

    age = st.number_input("Maternal Age", min_value=15, max_value=55, value=32, step=1)
    ntmom_1st = st.number_input("NtMoM", min_value=0.1, max_value=5.0, value=1.0, key="ntmom_1st")
    nipt_result = st.selectbox("NIPT_results", options=[0, 1, 2], help="0 = Negative, 1 = Positive, 2 = Not reported")

    input_df_1st = pd.DataFrame([{
        "Age": age,
        "NtMoM": ntmom_1st,
        "NIPT_results": nipt_result
    }])

    @st.cache_resource
    def load_models(target_name):
        with open(f"{MODEL_DIR}/bootstrapped_lasso_models_{target_name}.pkl", "rb") as f:
            return pickle.load(f)

    try:
        models = load_models(target_1st)
        predictions = [model.predict_proba(input_df_1st)[0][1] for model in models]

        predictions = np.array(predictions)
        mean_prob = np.mean(predictions)
        lower_ci = np.percentile(predictions, 2.5)
        upper_ci = np.percentile(predictions, 97.5)

        st.markdown(f"### Prediction Result for **{target_1st}**")
        st.write(f"**Mean predicted probability:** {mean_prob:.3f}")
        st.write(f"**95% Confidence Interval:** ({lower_ci:.3f}, {upper_ci:.3f})")

        fig, ax = plt.subplots()
        ax.hist(predictions, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(mean_prob, color='red', linestyle='--', label='Your Predicted Risk')
        ax.set_title(f"Bootstrapped Predictions for {target_1st}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading 1st trimester models or predicting: {e}")

# ------------------- PANEL 2: Second Trimester CH-Resolved Calculator -------------------
with col2:
    st.subheader("Second Trimester CH Resolution Calculator")
    st.caption("Trained on 65 non-aneuploidy cases with known CH resolution status")

    target_2nd = st.selectbox("Select 2nd trimester outcome:", ["Anomaly", "Livebirth", "OtherChroms"], key="2nd_trimester_selector")
    ntmom_2nd = st.number_input("NtMoM", min_value=0.1, max_value=5.0, value=1.0, key="ntmom_2nd")
    resolved_ch = st.selectbox("ResolvedCH", options=[0, 1], format_func=lambda x: f"{x} ({'Unresolved' if x == 0 else 'Resolved'})")

    input_df_2nd = pd.DataFrame([{
        "NtMoM": ntmom_2nd,
        "ResolvedCH": resolved_ch
    }])

    @st.cache_resource
    def load_ch_model(target_name):
        path_map = {
            "Anomaly": "bootstrapped_MLPClassifier_models_Anomaly.pkl",
            "Livebirth": "bootstrapped_BernoulliNB_models_Livebirth.pkl",
            "OtherChroms": "bootstrapped_SVC_models_OtherChroms.pkl"
        }
        with open(path_map[target_name], "rb") as f:
            return pickle.load(f)

    try:
        ch_models = load_ch_model(target_2nd)
        predictions = [model.predict_proba(input_df_2nd)[0][1] for model in ch_models]

        predictions = np.array(predictions)
        mean_prob = np.mean(predictions)
        lower_ci = np.percentile(predictions, 2.5)
        upper_ci = np.percentile(predictions, 97.5)

        st.markdown(f"### Prediction Result for **{target_2nd}**")
        st.write(f"**Mean predicted probability:** {mean_prob:.3f}")
        st.write(f"**95% Confidence Interval:** ({lower_ci:.3f}, {upper_ci:.3f})")

        fig, ax = plt.subplots()
        ax.hist(predictions, bins=30, alpha=0.7, color="lightgreen", edgecolor="black")
        ax.axvline(mean_prob, color='darkgreen', linestyle='--', label='Your Predicted Risk')
        ax.set_title(f"Bootstrapped Predictions for {target_2nd}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading 2nd trimester models or predicting: {e}")
