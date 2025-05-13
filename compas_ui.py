import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recidivism Risk Predictor", layout="wide")
st.title("🧠 Recidivism Risk Prediction Tool")

if "step" not in st.session_state:
    st.session_state.step = 1
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

model = joblib.load("recidivism_xgb_pipeline.pkl")
labels = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}

def get_confidence_message(p, pred):
    if p >= 0.85:
        return f"{labels[pred]} (✔️ High Confidence)"
    elif p >= 0.6:
        return f"{labels[pred]} (⚠️ Moderate Confidence)"
    else:
        return f"{labels[pred]} (❗ Low Confidence – Other classes are also possible)"

def plot_probabilities(proba):
    fig, ax = plt.subplots()
    bars = ax.bar(labels.values(), proba, color=["green", "orange", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Risk Probability Distribution")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.0%}", ha='center', va='bottom')
    st.pyplot(fig)

# === Step Navigation ===
if st.session_state.step == 1:
    st.markdown("""
    This tool estimates the likelihood of reoffending based on socio-demographic and behavioral data.
    Please fill out the form and click **Predict**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        age_group = st.selectbox("Age Group", ["12-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        race_ethnicity = st.selectbox("Ethnicity", ["Turk", "Other"])

        if age_group == "12-14":
            education_options = ["Illiterate", "Literate without schooling", "Primary School", "Middle School"]
            marital_options = ["Single"]
            employment_options = ["Student", "Employed"]
        elif age_group == "15-17":
            education_options = ["Illiterate", "Literate without schooling", "Primary School", "Middle School", "High School"]
            marital_options = ["Single"]
            employment_options = ["Student", "Employed"]
        elif age_group in ["18-24", "25-34"]:
            education_options = ["Illiterate", "Literate without schooling", "Primary School",
                                 "Middle School", "High School", "Bachelor’s Degree"]
            marital_options = ["Single", "Married", "Divorced"]
            employment_options = ["Student", "Employed", "Unemployed"]
        else:
            education_options = ["Illiterate", "Literate without schooling", "Primary School",
                                 "Middle School", "High School", "Bachelor’s Degree", "Master/PhD"]
            marital_options = ["Single", "Married", "Divorced"]
            employment_options = ["Employed", "Unemployed", "Student", "Retired"]

        education_level = st.selectbox("Education Level", education_options, help="Highest level of education completed.")
        marital_status = st.selectbox("Marital Status", marital_options, help="Current legal marital status.")

    with col2:
        employment_status = st.selectbox("Employment Status", employment_options, help="Current work situation.")
        housing_status = st.selectbox("Housing Status", ["Houseowner", "Rent", "Homeless"],
                                      help="Current living situation.")

        prior_convictions = st.slider("Number of Prior Convictions", 0, 20, 0,
                                      help="Number of past criminal convictions (excluding juvenile).")
        juvenile_convictions = st.slider("Number of Juvenile Convictions", 0, 10, 0,
                                         help="Number of criminal convictions before adulthood.")

    with col3:
        no_prior = (prior_convictions == 0 and juvenile_convictions == 0)
        prior_probation_violation = st.checkbox(
            "Probation Violation?", value=False, disabled=no_prior,
            help="Has the individual violated any terms during probation or parole?"
        )
        prior_incarceration = st.checkbox(
            "Has Been Incarcerated?", value=False, disabled=no_prior,
            help="Has the individual ever served time in jail or prison?"
        )
        if no_prior:
            st.caption("ℹ️ Usually, no prior convictions means no probation violation or incarceration.")

        is_child = age_group == "12-14"
        has_dependents = st.checkbox(
            "Has Dependents?", value=False, disabled=is_child,
            help="Responsible for supporting one or more dependents (e.g., children)."
        )
        if is_child:
            st.caption("ℹ️ Individuals aged 12-14 are assumed not to have dependents.")

        substance_abuse_history = st.checkbox("Substance Abuse History?", value=False,
            help="History of problematic use of drugs or alcohol.")
        mental_health_issues = st.checkbox("Mental Health Issues?", value=False,
            help="Any diagnosed or suspected psychological disorder.")
        gang_affiliation = st.checkbox("Gang Affiliation?", value=False,
            help="Known connection to gang activity.")
        aggression_history = st.checkbox("Aggression History?", value=False,
            help="Past behavior involving physical violence or aggression.")
        compliance_history = st.checkbox("Compliant with Rules?", value=False,
            help="Known to follow supervision rules or conditions.")
        motivation_to_change = st.checkbox("Motivated to Change?", value=False,
            help="Has demonstrated desire to rehabilitate or improve behavior.")
        stable_employment_past = st.checkbox("Stable Employment History?", value=False,
            help="Has held consistent jobs in the past.")
        positive_social_support = st.checkbox("Positive Social Support?", value=False,
            help="Has healthy social connections that discourage crime.")

    if st.button("🧮 Predict"):
        input_dict = {
            "age_group": age_group,
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "education_level": education_level,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "housing_status": housing_status,
            "has_dependents": has_dependents,
            "prior_convictions": prior_convictions,
            "juvenile_convictions": juvenile_convictions,
            "prior_probation_violation": prior_probation_violation,
            "prior_incarceration": prior_incarceration,
            "substance_abuse_history": substance_abuse_history,
            "mental_health_issues": mental_health_issues,
            "gang_affiliation": gang_affiliation,
            "aggression_history": aggression_history,
            "compliance_history": compliance_history,
            "motivation_to_change": motivation_to_change,
            "stable_employment_past": stable_employment_past,
            "positive_social_support": positive_social_support
        }

        df_input = pd.DataFrame([input_dict])
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]

        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        st.write(f"**Risk Level:** {get_confidence_message(proba[pred], pred)}")
        st.progress(int(proba[pred] * 100))

        with st.expander("📊 Show Class Probabilities"):
            plot_probabilities(proba)

        st.markdown("#### 📋 Interpretive Note")
        if proba[2] > 0.4:
            st.error("🔴 High likelihood of reoffending. Preventive intervention is strongly recommended.")
        elif proba[1] > 0.4:
            st.warning("🟡 Medium risk. Behavioral observation and support may help.")
        elif proba[0] > 0.8:
            st.success("🟢 Low risk. Still, periodic monitoring is advised.")
        else:
            st.info("ℹ️ Risk level is ambiguous. Consider gathering more data.")

    if st.button("Next: Crime Type"):
        input_dict = {
            "age_group": age_group,
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "education_level": education_level,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "housing_status": housing_status,
            "has_dependents": has_dependents,
            "prior_convictions": prior_convictions,
            "juvenile_convictions": juvenile_convictions,
            "prior_probation_violation": prior_probation_violation,
            "prior_incarceration": prior_incarceration,
            "substance_abuse_history": substance_abuse_history,
            "mental_health_issues": mental_health_issues,
            "gang_affiliation": gang_affiliation,
            "aggression_history": aggression_history,
            "compliance_history": compliance_history,
            "motivation_to_change": motivation_to_change,
            "stable_employment_past": stable_employment_past,
            "positive_social_support": positive_social_support
        }
        df_input = pd.DataFrame([input_dict])
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        st.session_state.user_data["recidivism_input"] = input_dict
        st.session_state.user_data["risk_pred"] = int(pred)
        st.session_state.user_data["risk_proba"] = proba.tolist()
        st.session_state.step = 2
        st.rerun()

# Step 2 - Crime Type
elif st.session_state.step == 2:
    st.header("Step 2: Crime Type")
    crime_type = st.selectbox("Select the committed crime", ["Intentional Killing (TCK 82)"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next: Aggravating Circumstances"):
            st.session_state.user_data["crime_type"] = crime_type
            st.session_state.step = 3
            st.rerun()