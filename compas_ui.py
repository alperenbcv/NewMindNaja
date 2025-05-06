# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Modeli yükle
model = joblib.load("recidivism_model.pkl")

st.title("Sanık Risk Tahmin Aracı (Recidivism Predictor)")

# Giriş formu
with st.form("input_form"):
    age_group = st.selectbox("Yaş Grubu", ["12-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    gender = st.selectbox("Cinsiyet", ["Male", "Female"])
    race_ethnicity = st.selectbox("Etnik Grup", ["Turk", "Kurd", "Arab", "Other"])
    education_level = st.selectbox("Eğitim Düzeyi", ["Illiterate", "Literate without schooling", "Primary School",
                                                     "Middle School", "High School", "Bachelor’s Degree", "Master/PhD"])
    marital_status = st.selectbox("Medeni Hali", ["Single", "Married", "Divorced"])
    employment_status = st.selectbox("İstihdam Durumu", ["Employed", "Unemployed", "Student", "Retired"])
    housing_status = st.selectbox("Barınma Durumu", ["Houseowner", "Rent", "Homeless"])
    has_dependents = st.checkbox("Bakmakla Yükümlü Olduğu Kişi Var mı?")
    prior_convictions = st.slider("Önceki Sabıka Sayısı", 0, 20, 0)
    juvenile_convictions = st.slider("Çocukken Sabıka Sayısı", 0, 10, 0)
    prior_probation_violation = st.checkbox("Daha Önce Denetimli Serbestlik İhlali Var mı?")
    prior_incarceration = st.checkbox("Daha Önce Hapsedildi mi?")
    substance_abuse_history = st.checkbox("Madde Bağımlılığı Geçmişi Var mı?")
    mental_health_issues = st.checkbox("Ruhsal Sağlık Sorunu Var mı?")
    gang_affiliation = st.checkbox("Çete Bağlantısı Var mı?")
    aggression_history = st.checkbox("Saldırganlık Geçmişi Var mı?")
    compliance_history = st.checkbox("Kurallara Uyumsuzluk Geçmişi Var mı?")
    motivation_to_change = st.checkbox("Değişime Motivasyonu Var mı?")
    stable_employment_past = st.checkbox("Geçmişte İstikrarlı İşi Oldu mu?")
    positive_social_support = st.checkbox("Pozitif Sosyal Destek Var mı?")

    submitted = st.form_submit_button("Tahmin Et")

if submitted:
    # Giriş verisini bir dataframe'e çevir
    input_dict = {
        "age_group": [age_group],
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "education_level": [education_level],
        "marital_status": [marital_status],
        "employment_status": [employment_status],
        "housing_status": [housing_status],
        "has_dependents": [has_dependents],
        "prior_convictions": [prior_convictions],
        "juvenile_convictions": [juvenile_convictions],
        "prior_probation_violation": [prior_probation_violation],
        "prior_incarceration": [prior_incarceration],
        "substance_abuse_history": [substance_abuse_history],
        "mental_health_issues": [mental_health_issues],
        "gang_affiliation": [gang_affiliation],
        "aggression_history": [aggression_history],
        "compliance_history": [compliance_history],
        "motivation_to_change": [motivation_to_change],
        "stable_employment_past": [stable_employment_past],
        "positive_social_support": [positive_social_support]
    }

    input_df = pd.DataFrame(input_dict)

    # Modelin gördüğü feature set'ine göre one-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Eksik sütunları modelin beklediği şekilde tamamla
    # Önceden kullanılan CSV'nin dummy sütunlarına göre boş olanları 0'la doldur
    model_features = model.named_steps["clf"].n_features_in_  # eski sklearn sürümünde hata verebilir
    X_model = pd.read_csv("mock_data.csv")
    all_columns = pd.get_dummies(X_model.drop(columns="recidivism"), drop_first=True).columns
    input_df = input_df.reindex(columns=all_columns, fill_value=0)

    # Tahmin
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    label_map = {0: "Düşük Risk", 1: "Orta Risk", 2: "Yüksek Risk"}

    st.subheader("🔍 Tahmin Sonucu")
    st.write(f"**Risk Seviyesi:** {label_map[prediction]}")
    st.progress(int(prediction_proba[prediction] * 100))

    st.write("**Tüm sınıfların olasılıkları:**")
    for i, prob in enumerate(prediction_proba):
        st.write(f"{label_map[i]}: {prob:.2%}")
