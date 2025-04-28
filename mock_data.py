import pandas as pd
import numpy as np

# -------------------------------
# Kategoriler ve dağılımlar
# -------------------------------
genders = ["Male", "Female"]
gender_probs = [0.916, 0.084]

races = ["Turk", "Kurd", "Arab", "Other"]
race_probs = [0.75, 0.15, 0.05, 0.05]

education_levels = [
    "Illiterate", "Literate without schooling", "Primary School",
    "Middle School", "High School", "Bachelor’s Degree", "Master/PhD"
]
education_probs = [
    0.037, 0.092, 0.319, 0.289, 0.226, 0.035, 0.002
]

marital_statuses = ["Single", "Married", "Divorced"]
marital_probs = [0.5, 0.4, 0.1]

employment_statuses = ["Employed", "Unemployed", "Student", "Retired"]
employment_probs = [0.4, 0.3, 0.15, 0.15]

housing_statuses = ["Houseowner", "Rent", "Homeless"]
housing_probs = [0.5, 0.45, 0.05]

age_groups = ["12-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
age_probs = [0.0007, 0.0083, 0.1240, 0.3716, 0.2865, 0.1442, 0.0502, 0.0145]

yes_no = [True, False]

# -------------------------------
# Ağırlıklar
# -------------------------------
housing_recidivism_weight = {
    "Houseowner": 0.9,
    "Rent": 1.1,
    "Homeless": 2
}

gender_recidivism_weight = {
    "Male": 1.0,
    "Female": 0.9
}

race_recidivism_weight = {
    "Turk": 1.0,
    "Kurd": 1.1,
    "Arab": 1.1,
    "Other": 1.1
}

education_recidivism_weight = {
    "Illiterate": 1.5,
    "Literate without schooling": 1.4,
    "Primary School": 1.3,
    "Middle School": 1.2,
    "High School": 1.0,
    "Bachelor’s Degree": 0.8,
    "Master/PhD": 0.7
}

age_recidivism_weight = {
    "12-14": 1.8,
    "15-17": 1.6,
    "18-24": 1.5,
    "25-34": 1.2,
    "35-44": 1.0,
    "45-54": 0.9,
    "55-64": 0.8,
    "65+": 0.6
}

marital_status_weight = {
    "Single": 1.0,
    "Married": 0.9,
    "Divorced": 1.3
}

employment_status_weight = {
    "Employed": 0.9,
    "Unemployed": 1.2,
    "Student": 1,
    "Retired": 0.7
}

base_recidivism_rate = 0.45

# -------------------------------
# Veri üretimi
# -------------------------------
n_samples = 300
np.random.seed(42)

mock_data = pd.DataFrame({
    "age_group": np.random.choice(age_groups, n_samples, p=age_probs),
    "gender": np.random.choice(genders, n_samples, p=gender_probs),
    "race_ethnicity": np.random.choice(races, n_samples, p=race_probs),
    "education_level": np.random.choice(education_levels, n_samples, p=education_probs),
    "marital_status": np.random.choice(marital_statuses, n_samples, p=marital_probs),
    "employment_status": np.random.choice(employment_statuses, n_samples, p=employment_probs),
    "housing_status": np.random.choice(housing_statuses, n_samples, p=housing_probs),
    "has_dependents": np.random.choice(yes_no, n_samples, p=[0.5, 0.5]),
    "prior_convictions": np.random.poisson(1.5, n_samples),
    "juvenile_convictions": np.random.poisson(0.5, n_samples),
    "prior_violent_offenses": np.random.poisson(0.5, n_samples),
    "prior_probation_violation": np.random.choice(yes_no, n_samples, p=[0.3, 0.7]),
    "prior_incarceration": np.random.choice(yes_no, n_samples, p=[0.4, 0.6]),
    "mental_health_issues": np.random.choice(yes_no, n_samples, p=[0.1, 0.9]),
    "substance_abuse_history": np.random.choice(yes_no, n_samples, p=[0.1, 0.9]),
    "gang_affiliation": np.random.choice(yes_no, n_samples, p=[0.05, 0.95]),
    "aggression_history": np.random.choice(yes_no, n_samples, p=[0.3, 0.7]),
    "compliance_history": np.random.choice(yes_no, n_samples, p=[0.2, 0.8]),
    "motivation_to_change": np.random.choice(yes_no, n_samples, p=[0.7, 0.3]),
    "stable_employment_past": np.random.choice(yes_no, n_samples, p=[0.6, 0.4]),
    "stable_residence_past": np.random.choice(yes_no, n_samples, p=[0.7, 0.3]),
    "positive_social_support": np.random.choice(yes_no, n_samples, p=[0.6, 0.4])
})

# -------------------------------
# Recidivism hesaplama
# -------------------------------

def calculate_recidivism(row):
    housing_weight = housing_recidivism_weight.get(row["housing_status"], 1.0)
    gender_weight = gender_recidivism_weight.get(row["gender"], 1.0)
    race_weight = race_recidivism_weight.get(row["race_ethnicity"], 1.0)
    education_weight = education_recidivism_weight.get(row["education_level"], 1.0)
    age_weight = age_recidivism_weight.get(row["age_group"], 1.0)
    marital_weight = marital_status_weight.get(row["marital_status"], 1.0)
    employment_weight = employment_status_weight.get(row["employment_status"], 1.0)
    prior_offenses_factor = 1 + (row["prior_convictions"] * 0.05)
    juvenile_offenses_factor = 1 + (row["juvenile_convictions"] * 0.06)
    probation_violation_factor = 1.2 if row["prior_probation_violation"] else 1.0
    incarceration_factor = 1.2 if row["prior_incarceration"] else 1.0
    substance_abuse_factor = 1.3 if row["substance_abuse_history"] else 1.0
    mental_health_factor = 1.2 if row["mental_health_issues"] else 1.0
    gang_affiliation_factor = 1.3 if row["gang_affiliation"] else 1.0
    aggression_factor = 1.2 if row["aggression_history"] else 1.0
    motivation_factor = 0.8 if row["motivation_to_change"] else 1.0
    compliance_factor = 1.2 if not row["compliance_history"] else 1.0
    positive_social_support_factor = 0.8 if row["positive_social_support"] else 1.2
    dependents_factor = 0.9 if row["has_dependents"] else 1.1

    combined_weight = (housing_weight * gender_weight * race_weight * education_weight *
                       age_weight * marital_weight * employment_weight)
    combined_weight *= (prior_offenses_factor * juvenile_offenses_factor * probation_violation_factor *
                        incarceration_factor * substance_abuse_factor * mental_health_factor *
                        gang_affiliation_factor * aggression_factor * motivation_factor *
                        compliance_factor *
                        positive_social_support_factor * dependents_factor)

    personal_recidivism_probability = base_recidivism_rate * combined_weight
    personal_recidivism_probability = min(personal_recidivism_probability, 1.0)

    return int(np.random.rand() < personal_recidivism_probability)

mock_data["recidivism"] = mock_data.apply(calculate_recidivism, axis=1)

# -------------------------------
# Kaydet
# -------------------------------
mock_data.to_csv("mock_compas_data_weighted_full.csv", index=False)

print("Mock veri tüm 22 veri faktörü dahil edilerek başarıyla oluşturuldu!")
