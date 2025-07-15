import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
TARGET_OPTIONS = [
    'Composite', 'Anomaly', 'ChromsProb', 'SAB', 'Livebirth',
    "T21", "T18", "T13", "Turners", "OtherChroms"
]
MODEL_DIR = "."  # Folder where .pkl models are saved

# ------------------- USER INPUT -------------------
st.title("Prenatal Risk Prediction Calculator")

target = st.selectbox("Select the outcome to predict:", TARGET_OPTIONS)

st.markdown("### Input Patient Information")
age = st.number_input("Age", min_value=15.0, max_value=55.0, value=32.0)
ntmom = st.number_input("NtMoM", min_value=0.1, max_value=5.0, value=1.0)
nipt_result = st.selectbox("NIPT_results", options=[0, 1, 2], help="0=Negative, 1=Positive, 2=Inconclusive/Missing")

user_input_df = pd.DataFrame([{
    "Age": age,
    "NtMoM": ntmom,
    "NIPT_results": nipt_result
}])

# ------------------- PREDICTION -------------------
@st.cache_resource
def load_models(target_name):
    with open(f"{MODEL_DIR}/bootstrapped_lasso_models_{target_name}.pkl", "rb") as f:
        return pickle.load(f)

if st.button("Predict Risk"):
    try:
        models = load_models(target)
        predictions = []
        for model in models:
            pred = model.predict_proba(user_input_df)[0][1]  # probability of class 1
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_prob = np.mean(predictions)
        lower_ci = np.percentile(predictions, 2.5)
        upper_ci = np.percentile(predictions, 97.5)

        # ------------------- RESULTS -------------------
        st.markdown(f"## Prediction Result for **{target}**")
        st.write(f"**Mean predicted probability:** {mean_prob:.3f}")
        st.write(f"**95% Confidence Interval:** ({lower_ci:.3f}, {upper_ci:.3f})")

        # ------------------- HISTOGRAM -------------------
        fig, ax = plt.subplots()
        ax.hist(predictions, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(mean_prob, color='red', linestyle='--', label='Your Predicted Risk')
        ax.set_title(f"Bootstrapped Predictions for {target}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading models or predicting: {e}")
