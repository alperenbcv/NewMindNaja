# mock_data_generator.py
import pandas as pd
import numpy as np

def generate_mock_data(seed=42, n_samples=3000):
    np.random.seed(seed)
    genders = ["Male","Female"]; gender_p=[0.916,0.084]
    races = ["Turk","Kurd","Arab","Other"]; race_p=[0.75,0.15,0.05,0.05]
    ed_lvls = ["Illiterate","Literate without schooling","Primary School","Middle School","High School","Bachelor’s Degree","Master/PhD"]
    ed_p = [0.037,0.092,0.319,0.289,0.226,0.035,0.002]
    maritals = ["Single","Married","Divorced"]; mar_p=[0.5,0.4,0.1]
    emps = ["Employed","Unemployed","Student","Retired"]; emp_p=[0.4,0.3,0.15,0.15]
    housings = ["Houseowner","Rent","Homeless"]; house_p=[0.4,0.55,0.05]
    ages = ["12-14","15-17","18-24","25-34","35-44","45-54","55-64","65+"]
    age_p=[0.0007,0.0083,0.1240,0.3716,0.2865,0.1442,0.0502,0.0145]
    yesno=[True,False]

    housing_w = {"Houseowner":0.9,"Rent":1.0,"Homeless":2}
    gender_w  = {"Male":1.0,"Female":0.7}
    race_w    = {"Turk":1.0,"Kurd":1.1,"Arab":1.1,"Other":1.1}
    ed_w      = {"Illiterate":1.5,"Literate without schooling":1.4,
                 "Primary School":1.3,"Middle School":1.2,"High School":1.0,
                 "Bachelor’s Degree":0.8,"Master/PhD":0.7}
    age_w     = {"12-14":1.6,"15-17":1.5,"18-24":1.3,"25-34":1.2,
                 "35-44":1.0,"45-54":0.9,"55-64":0.8,"65+":0.6}
    marital_w = {"Single":1.0,"Married":0.95,"Divorced":1.2}
    emp_w     = {"Employed":0.9,"Unemployed":1.2,"Student":1.0,"Retired":0.7}
    base_rate = 0.3

    df = pd.DataFrame({
        "age_group": np.random.choice(ages, n_samples, p=age_p),
        "gender":    np.random.choice(genders, n_samples, p=gender_p),
        "race_ethnicity": np.random.choice(races, n_samples, p=race_p),
        "education_level": np.random.choice(ed_lvls, n_samples, p=ed_p),
        "marital_status":  np.random.choice(maritals, n_samples, p=mar_p),
        "employment_status": np.random.choice(emps, n_samples, p=emp_p),
        "housing_status":   np.random.choice(housings, n_samples, p=house_p),
        "has_dependents":   np.random.choice(yesno, n_samples, p=[0.6,0.4]),
        "prior_convictions":      np.random.poisson(3, n_samples),
        "juvenile_convictions":   np.random.poisson(1.5, n_samples),
        "prior_probation_violation": np.random.choice(yesno, n_samples, p=[0.3,0.7]),
        "prior_incarceration":       np.random.choice(yesno, n_samples, p=[0.4,0.6]),
        "substance_abuse_history":   np.random.choice(yesno, n_samples, p=[0.3,0.7]),
        "mental_health_issues":      np.random.choice(yesno, n_samples, p=[0.25,0.75]),
        "gang_affiliation":          np.random.choice(yesno, n_samples, p=[0.03,0.97]),
        "aggression_history":        np.random.choice(yesno, n_samples, p=[0.3,0.7]),
        "compliance_history":        np.random.choice(yesno, n_samples, p=[0.2,0.8]),
        "motivation_to_change":      np.random.choice(yesno, n_samples, p=[0.6,0.4]),
        "stable_employment_past":    np.random.choice(yesno, n_samples, p=[0.55,0.45]),
        "positive_social_support":   np.random.choice(yesno, n_samples, p=[0.6,0.4])
    })

    def calc_prob(r):
        w = (housing_w[r.housing_status]*gender_w[r.gender]*race_w[r.race_ethnicity]*
             ed_w[r.education_level]*age_w[r.age_group]*marital_w[r.marital_status]*emp_w[r.employment_status])
        w *= (1+0.05*r.prior_convictions)*(1+0.07*r.juvenile_convictions)
        w *= 1.2 if r.prior_probation_violation else 1.0
        w *= 1.2 if r.prior_incarceration else 0.9
        w *= 1.2 if r.substance_abuse_history else 0.9
        w *= 1.2 if r.mental_health_issues else 1.0
        w *= 1.3 if r.gang_affiliation else 1.0
        w *= 1.2 if r.aggression_history else 0.9
        w *= 0.9 if r.motivation_to_change else 1.0
        w *= 1.2 if not r.compliance_history else 1.0
        w *= 0.9 if r.stable_employment_past else 1.1
        w *= 0.9 if r.positive_social_support else 1.1
        w *= 0.9 if r.has_dependents else 1.1
        return min(base_rate * w, 1.0)

    df["recidivism_prob"] = df.apply(calc_prob, axis=1)

    def assign_cls(p):
        th = [0.45, 0.7]
        if p < th[0]:
            return np.random.choice([0,1], p=[0.9,0.1])
        elif p < th[1]:
            return np.random.choice([1,2], p=[0.8,0.2])
        else:
            return np.random.choice([2,1], p=[0.85,0.15])

    df["recidivism"] = df["recidivism_prob"].apply(assign_cls)
    return df.drop(columns="recidivism_prob")

if __name__ == "__main__":
    df = generate_mock_data(seed=42)
    df.to_csv("mock_data.csv", index=False)
    print("✔ mock_data.csv dosyası oluşturuldu.")
