import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sanık Risk Tahmini", layout="wide")
st.title("🧠 Sanık Risk Tahmin Aracı")
st.markdown("""
Bu araç, bir sanığın yeniden suç işleme olasılığını tahmin eder. 
Lütfen aşağıdaki bilgileri doldurun ve "Tahmin Et" butonuna basın.
""")

# Modeli yükle
model = joblib.load("recidivism_xgb_pipeline.pkl")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age_group = st.selectbox("Yaş Grubu", ["12-14","15-17","18-24","25-34","35-44","45-54","55-64","65+"])
        gender = st.selectbox("Cinsiyet", ["Male","Female"])
        race_ethnicity = st.selectbox("Etnik Grup", ["Turk","Kurd","Arab","Other"])
        education_level = st.selectbox("Eğitim Düzeyi", [
            "Illiterate","Literate without schooling","Primary School",
            "Middle School","High School","Bachelor’s Degree","Master/PhD"
        ])
        marital_status = st.selectbox("Medeni Hali", ["Single","Married","Divorced"])
    with col2:
        employment_status = st.selectbox("İstihdam Durumu", ["Employed","Unemployed","Student","Retired"])
        housing_status = st.selectbox("Barınma Durumu", ["Houseowner","Rent","Homeless"])
        has_dependents = st.checkbox("Bakmakla Yükümlü Var mı?", value=False)
        prior_convictions = st.slider("Önceki Sabıka Sayısı", 0, 20, 0)
        juvenile_convictions = st.slider("Çocuk Sabıka Sayısı", 0, 10, 0)
    with col3:
        prior_probation_violation = st.checkbox("Denetimli Serbestlik İhlali?", value=False)
        prior_incarceration = st.checkbox("Hapsedilmiş mi?", value=False)
        substance_abuse_history = st.checkbox("Madde Bağımlılığı?", value=False)
        mental_health_issues = st.checkbox("Ruhsal Sağlık Sorunu?", value=False)
        gang_affiliation = st.checkbox("Çete Bağlantısı?", value=False)
        aggression_history = st.checkbox("Saldırganlık Geçmişi?", value=False)
        compliance_history = st.checkbox("Kurallara Uyum?", value=False)
        motivation_to_change = st.checkbox("Değişime Motivasyon?", value=False)
        stable_employment_past = st.checkbox("İstikrarlı İş Geçmişi?", value=False)
        positive_social_support = st.checkbox("Pozitif Sosyal Destek?", value=False)

    submitted = st.form_submit_button("🧮 Tahmin Et")

if submitted:
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

    labels = {0: "🟢 Düşük Risk", 1: "🟡 Orta Risk", 2: "🔴 Yüksek Risk"}

    st.markdown("---")
    st.subheader("🔍 Tahmin Sonucu")
    st.write(f"**Risk Seviyesi:** {labels[pred]}")
    st.progress(int(proba[pred] * 100))

    st.markdown("#### 🔢 Tüm Sınıf Olasılıkları")
    for i, p in enumerate(proba):
        st.write(f"{labels[i]}: `{p:.2%}`")

    st.markdown("✅ Tahmin başarıyla tamamlandı.")

