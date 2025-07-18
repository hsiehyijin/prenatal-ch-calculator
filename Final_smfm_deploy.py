import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
MODEL_DIR = "."  # Folder where .pkl models are saved

# ------------------- PAGE SETUP -------------------
st.set_page_config(layout="wide")
st.image("lab_logo.png", width=200)
st.title("Prenatal Risk Prediction Calculator")
st.caption("Reference: Wang et al. Non-Invasive Prenatal Testing Results, Nuchal Translucency Size, and Second Trimester Resolution Modify First Trimester Cystic Hygroma Outcomes, 2025; Prenatal Diagnosis. DOI: 10.1002/pd.6791 ")

# ------------------- USER INPUT COLUMNS -------------------
col1, col2 = st.columns(2)

# ------------------- PANEL 1: First Trimester Calculator -------------------
with col1:
    st.markdown(
    """
    <div style='background-color: #e0f7fa; padding: 10px; border-radius: 5px;'>
        <span style='font-size:28px; font-weight:600;'>First Trimester Calculator</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.caption("Trained on 259 cases with cystic hygroma (CH)")

    age_1st = st.number_input("Maternal Age", min_value=15, max_value=50, value=32, key="age1")
    nt_1st = st.number_input("NT (mm)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, key="nt1")
    crl_1st = st.number_input("CRL (mm)", min_value=5.0, max_value=100.0, value=50.0, step=0.1, key="crl1")
    nipt_1st = st.selectbox("NIPT result:", ["Normal", "Abnormal", "Not Reported"], key="nipt1")

    st.markdown("<div style='background-color: #e0f7fa; padding: 10px; border-radius: 5px;'><b style='color: black;'>Select 1st trimester outcome:</b></div>", unsafe_allow_html=True)
   
    # Map display labels to model keys
    t1_target_display = {
        'Composite adverse fetal outcome': 'Composite',
        'Other structural anomaly': 'Anomaly',
        'Any genetic diagnoses': 'Common aneuploidy (T21, T13, T18, Monosomy X)',
        'SAB/IUFD': 'SAB',
        'Livebirth': 'Livebirth',
        'Trisomy 21 (Down syndrome)': 'Trisomy 21 (Down syndrome)',
        'Trisomy 18': 'Trisomy 18',
        'Trisomy 13': 'Trisomy 13',
        'Monosomy X (Turner syndrome)': 'Monosomy X (Turner syndrome)',
        'Other genetic diagnoses': 'Other genetic diagnoses'
    }

    # Use display names in dropdown, then map to internal key
    target_1st_display = st.selectbox("", list(t1_target_display.keys()), key="1st_trimester_selector")
    target_1st = t1_target_display[target_1st_display]

    if st.button("Predict Outcome for T1"):
        ntmom_cal_1st = nt_1st / (0.437 + 0.01969 * crl_1st)
        X_input = pd.DataFrame([{
            'Age': age_1st,
            'NTMoM_cal': ntmom_cal_1st,
            'NIPT_results': {"Normal": 0, "Abnormal": 1, "Not Reported": 2}[nipt_1st]
        }])

        try:
            with open(f"{MODEL_DIR}/bootstrapped_lasso_models_{target_1st}.pkl", "rb") as f:
                models = pickle.load(f)

            # Get prediction probabilities across all bootstraps
            probs = [m.predict_proba(X_input)[0][1] for m in models]
            lower = np.percentile(probs, 2.5)
            upper = np.percentile(probs, 97.5)
            prob = np.mean(probs)

            st.markdown("**\nPredicted Risk:**")
            st.markdown(f"### :orange[{round(prob * 100, 2)}% chance of {target_1st_display}]")
            st.markdown(f"95% CI: {round(lower * 100, 2)}% – {round(upper * 100, 2)}%")

            fig, ax = plt.subplots()
            ax.hist(probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(prob, color='red', linestyle='dashed', linewidth=2)
            ax.set_title("Bootstrapped Prediction Distribution")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Model error: {e}")

# ------------------- PANEL 2: Second Trimester Calculator -------------------
with col2:
    st.markdown(
    """
    <div style='background-color: #fff1e2; padding: 10px; border-radius: 5px;'>
        <span style='font-size:28px; font-weight:600;'>Second Trimester Calculator</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.caption("Trained on 66 non-aneuploidy cases with CH in second trimester")

    nt_2nd = st.number_input("First Trimester NT (mm)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, key="nt2")
    crl_2nd = st.number_input("CRL at time of NT (mm)", min_value=5.0, max_value=100.0, value=50.0, step=0.1, key="crl2")
    resolved_ch = st.selectbox("Resolved CH?", ["Yes", "No"], key="resolvedCH")

    st.markdown("<div style='background-color: #fff1e2; padding: 10px; border-radius: 5px;'><b style='color: black;'>Select 2nd trimester outcome:</b></div>", unsafe_allow_html=True)
        # Display-to-key mapping for T2 outcomes
    t2_target_display = {
        'Other structural anomaly': 'Anomaly',
        'Livebirth': 'Livebirth',
        'Other genetic diagnoses': 'Other genetic diagnoses'
    }

    target_2nd_display = st.selectbox("", list(t2_target_display.keys()), key="2nd_trimester_selector")
    target_2nd = t2_target_display[target_2nd_display]  # used for model file lookup

    if st.button("Predict Outcome for T2"):
        ntmom_cal_2nd = nt_2nd / (0.437 + 0.01969 * crl_2nd)
        X_input = pd.DataFrame([{
            'NTMoM_cal': ntmom_cal_2nd,
            'ResolvedCH': {"Yes": 1, "No": 0}[resolved_ch]
        }])

        model_mapping = {
            'Anomaly': 'bootstrapped_MLP_models_Anomaly.pkl',
            'Livebirth': 'bootstrapped_Lasso_Logistic_Regression_models_Livebirth.pkl',
            'Other genetic diagnoses': 'bootstrapped_Random_Forest_models_Other genetic diagnoses.pkl'
        }

        try:
            model_file = model_mapping[target_2nd]
            with open(f"{MODEL_DIR}/{model_file}", "rb") as f:
                models = pickle.load(f)

            probs = [m.predict_proba(X_input)[0][1] for m in models]
            lower = np.percentile(probs, 2.5)
            upper = np.percentile(probs, 97.5)
            prob = np.mean(probs)

            st.markdown("**\nPredicted Risk:**")
            st.markdown(f"### :orange[{round(prob * 100, 2)}% chance of {target_2nd_display}]")
            st.markdown(f"95% CI: {round(lower * 100, 2)}% – {round(upper * 100, 2)}%")

            fig, ax = plt.subplots()
            ax.hist(probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(prob, color='red', linestyle='dashed', linewidth=2)
            ax.set_title("Bootstrapped Prediction Distribution")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Model error: {e}")
