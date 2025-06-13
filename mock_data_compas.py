import pandas as pd
import numpy as np
from scipy.special import expit

# Etki katsayıları
age_w = {"12-14": 5.0, "15-17": 4.5, "18-24": 3.0, "25-34": 1.2,
         "35-44": 1.0, "45-54": 0.8, "55-64": 0.6, "65+": 0.4}
age_probs = [0.0007, 0.0083, 0.1240, 0.3716, 0.2865, 0.1442, 0.0502, 0.0145]

gender_w = {"Male": 1.0, "Female": 0.6}
race_w = {"Turk": 1.0, "Other": 1.2}

ed_w = {
    "Illiterate": 3.0,
    "Literate without schooling": 2.0,
    "Primary School": 1.5,
    "Middle School": 1.2,
    "High School": 1.0,
    "Bachelor’s Degree": 0.8,
    "Master/PhD": 0.5
}
education_probs = [0.037, 0.092, 0.319, 0.289, 0.226, 0.035, 0.002]

marital_w = {"Single": 1.0, "Married": 0.8, "Divorced": 1.3}
emp_w = {"Employed": 0.7, "Unemployed": 2.0, "Student": 1.2, "Retired": 0.6}
housing_w = {"Houseowner": 0.8, "Rent": 1.0, "Homeless": 2.5}

# Risk skoru
def calc_prob(r):
    s = -5
    s += np.log(age_w[r.age_group])
    s += np.log(gender_w[r.gender])
    s += np.log(race_w[r.race_ethnicity])
    s += np.log(ed_w[r.education_level])
    s += np.log(marital_w[r.marital_status])
    s += np.log(emp_w[r.employment_status])
    s += np.log(housing_w[r.housing_status])
    s += 0.5 * r.prior_convictions
    s += 0.7 * r.juvenile_convictions
    s += 0.7 if r.prior_probation_violation else 0
    s += 0.6 if r.prior_incarceration else 0
    s += 0.6 if r.substance_abuse_history else 0
    s += 0.5 if r.mental_health_issues else 0
    s += 1.2 if r.gang_affiliation else 0
    s += 0.8 if r.aggression_history else 0
    s += -0.5 if r.motivation_to_change else 0
    s += 0.4 if not r.compliance_history else -0.2
    s += -0.5 if r.stable_employment_past else 0.4
    s += -0.5 if r.positive_social_support else 0.4
    s += -0.3 if r.has_dependents else 0.2
    return expit(s)

# Riskli çocuk ekle
def add_high_risk_profiles(df, n=500):
    rng = np.random.default_rng(42)
    risky_list = []

    for _ in range(n):
        risky_kids = {
            "age_group": rng.choice(["12-14","15-17"]),
            "gender": rng.choice(["Male","Female"]),
            "race_ethnicity": rng.choice(["Turk","Other"]),
            "education_level": rng.choice(["Primary School","Illiterate","Literate without schooling"]),
            "marital_status": "Single",
            "employment_status": rng.choice(["Student","Employed"]),
            "housing_status": rng.choice(["Rent","Houseowner","Homeless"]),
            "has_dependents": rng.choice([True,False],p=[0.25, 0.75]),
            "prior_convictions": 0,
            "juvenile_convictions": rng.choice([0,1,2,3]),
            "prior_probation_violation": rng.choice([True,False],p=[0.25, 0.75]),
            "prior_incarceration": rng.choice([True,False],p=[0.25, 0.75]),
            "substance_abuse_history": rng.choice([True,False],p=[0.15, 0.85]),
            "mental_health_issues": False,
            "gang_affiliation": rng.choice([True,False],p=[0.15, 0.85]),
            "aggression_history": rng.choice([True,False],p=[0.25, 0.75]),
            "compliance_history": rng.choice([True,False],p=[0.50, 0.50]),
            "motivation_to_change": rng.choice([True,False],p=[0.50, 0.50]),
            "stable_employment_past": False,
            "positive_social_support": rng.choice([True,False],p=[0.25, 0.75])
        }
        risky_list.append(risky_kids)

    risky_df = pd.DataFrame(risky_list)
    return pd.concat([df, risky_df], ignore_index=True)


# Veri üretimi
def generate_mock_data(seed=42, n_samples=3000):
    rng = np.random.default_rng(seed)
    rows = []

    age_keys = list(age_w.keys())
    ed_keys = list(ed_w.keys())
    marital_keys = list(marital_w.keys())
    emp_keys = list(emp_w.keys())
    house_keys = list(housing_w.keys())

    for _ in range(n_samples):
        age_group = rng.choice(age_keys, p=age_probs)
        base = {
            "age_group": age_group,
            "gender": rng.choice(["Male", "Female"], p=[0.75, 0.25]),
            "race_ethnicity": rng.choice(["Turk", "Other"], p=[0.75, 0.25]),
            "housing_status": rng.choice(house_keys, p=[0.4, 0.5, 0.1]),
            "has_dependents": rng.choice([True, False]),
            "prior_probation_violation": rng.choice([True, False], p=[0.25, 0.75]),
            "prior_incarceration": rng.choice([True, False]),
            "substance_abuse_history": rng.choice([True, False]),
            "mental_health_issues": rng.choice([True, False], p=[0.25, 0.75]),
            "gang_affiliation": rng.choice([True, False], p=[0.15, 0.85]),
            "aggression_history": rng.choice([True, False]),
            "compliance_history": rng.choice([True, False]),
            "motivation_to_change": rng.choice([True, False]),
            "stable_employment_past": rng.choice([True, False]),
            "positive_social_support": rng.choice([True, False]),
        }

        if age_group in ["12-14", "15-17"]:
            base.update({
                "education_level": rng.choice(["Illiterate", "Literate without schooling", "Primary School", "Middle School"]),
                "marital_status": "Single",
                "employment_status": rng.choice(["Student", "Employed"]),
                "prior_convictions": 0,
                "juvenile_convictions": rng.integers(1, 4)
            })
        else:
            base.update({
                "education_level": rng.choice(ed_keys, p=education_probs),
                "marital_status": rng.choice(marital_keys, p=[0.5, 0.4, 0.1]),
                "employment_status": rng.choice(emp_keys),
                "prior_convictions": rng.poisson(3),
                "juvenile_convictions": rng.poisson(2)
            })

        rows.append(base)

    df = pd.DataFrame(rows)
    df = add_high_risk_profiles(df, n=300)
    df["recidivism_prob"] = df.apply(calc_prob, axis=1)

    # Sınıf etiketleri ata
    df = df.sort_values("recidivism_prob").reset_index(drop=True)
    n = len(df)
    df.loc[:int(n*0.33), "recidivism"] = 0
    df.loc[int(n*0.33):int(n*0.66), "recidivism"] = 1
    df.loc[int(n*0.66):, "recidivism"] = 2
    df["recidivism"] = df["recidivism"].astype(int)
    return df.drop(columns="recidivism_prob")

# Raporlama
def print_young_high_risk_cases(df):
    f = df[(df["age_group"] == "12-14") & (df["recidivism"] >= 1)]
    print(f"\n🧒 12-14 yaş grubunda orta/yüksek risk alan birey sayısı: {len(f)}")
    print(f.head(10).to_string(index=False))


# Sayım
    oniki_ondort= df[(df["age_group"] == "12-14")]
    onbes_onyedi= df[(df["age_group"] == "15-17")]
    deli = df[(df["mental_health_issues"] == True)]
    print(f"Total 12-14 yaş: {len(oniki_ondort)}")
    print(f"Total 15-17 yaş: {len(onbes_onyedi)}")
    print(f"Total deli: {len(deli)}")

# Çalıştır
if __name__ == "__main__":
    df = generate_mock_data()
    print(df["recidivism"].value_counts(normalize=True).rename("sınıf oranı"))
    print_young_high_risk_cases(df)
    df.to_csv("mock_data.csv", index=False)
    print("\n✅ mock_data.csv dosyasına yazıldı.")
