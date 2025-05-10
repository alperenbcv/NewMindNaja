import streamlit as st
import pandas as pd
import joblib

# Pipeline'ı yükle (içinde preproc+SMOTE+MLP var)
model = joblib.load("recidivism_logreg_pipeline.pkl")

st.title("Sanık Risk Tahmin Aracı")

with st.form("input_form"):
    age_group = st.selectbox("Yaş Grubu", ["12-14","15-17","18-24","25-34","35-44","45-54","55-64","65+"])
    gender = st.selectbox("Cinsiyet", ["Male","Female"])
    race_ethnicity = st.selectbox("Etnik Grup", ["Turk","Kurd","Arab","Other"])
    education_level = st.selectbox("Eğitim Düzeyi", [
        "Illiterate","Literate without schooling","Primary School",
        "Middle School","High School","Bachelor’s Degree","Master/PhD"
    ])
    marital_status = st.selectbox("Medeni Hali", ["Single","Married","Divorced"])
    employment_status = st.selectbox("İstihdam Durumu", ["Employed","Unemployed","Student","Retired"])
    housing_status = st.selectbox("Barınma Durumu", ["Houseowner","Rent","Homeless"])
    has_dependents = st.checkbox("Bakmakla Yükümlü Var mı?")
    prior_convictions = st.slider("Önceki Sabıka Sayısı", 0, 20, 0)
    juvenile_convictions = st.slider("Çocuk Sabıka Sayısı", 0, 10, 0)
    prior_probation_violation = st.checkbox("Daha Önce Denetimli Serbestlik İhlali?")
    prior_incarceration = st.checkbox("Daha Önce Hapsedildi mi?")
    substance_abuse_history = st.checkbox("Madde Bağımlılığı Geçmişi?")
    mental_health_issues = st.checkbox("Ruhsal Sağlık Sorunu?")
    gang_affiliation = st.checkbox("Çete Bağlantısı?")
    aggression_history = st.checkbox("Saldırganlık Geçmişi?")
    compliance_history = st.checkbox("Kurallara Uyumsuzluk?")
    motivation_to_change = st.checkbox("Değişime Motivasyon?")
    stable_employment_past = st.checkbox("Geçmişte İstikrarlı Çalışma?")
    positive_social_support = st.checkbox("Pozitif Sosyal Destek?")

    submitted = st.form_submit_button("Tahmin Et")

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
    labels = {0:"Düşük Risk",1:"Orta Risk",2:"Yüksek Risk"}

    st.subheader("🔍 Tahmin Sonucu")
    st.write(f"**Risk Seviyesi:** {labels[pred]}")
    st.progress(int(proba[pred]*100))
    st.write("**Tüm sınıfların olasılıkları:**")
    for i, p in enumerate(proba):
        st.write(f"{labels[i]}: {p:.2%}")
