import pandas as pd
import numpy as np
from scipy.special import expit  # sigmoid

def generate_mock_data(seed=42, n_samples=3000):
    np.random.seed(seed)

    # Kategoriler & olasılıklar
    ages = ["12-14","15-17","18-24","25-34","35-44","45-54","55-64","65+"]
    age_p = [1/8]*8
    genders = ["Male","Female"]; gender_p = [0.5,0.5]
    races = ["Turk","Kurd","Arab","Other"]; race_p = [1/4]*4
    ed_lvls = ["Illiterate","Literate without schooling","Primary School","Middle School","High School","Bachelor’s Degree","Master/PhD"]
    ed_p = [1/7]*7
    maritals = ["Single","Married","Divorced"]; mar_p = [1/3]*3
    emps = ["Employed","Unemployed","Student","Retired"]; emp_p = [1/4]*4
    housings = ["Houseowner","Rent","Homeless"]; house_p = [1/3]*3
    yesno = [True, False]

    # Ağırlıklar (özellikle yaş grubunu büyüttük)
    age_w = {
        "12-14":2.0, "15-17":1.8, "18-24":1.5, "25-34":1.2,
        "35-44":1.0, "45-54":0.8, "55-64":0.6, "65+":0.4
    }
    gender_w  = {"Male":1.0,"Female":0.7}
    race_w    = {"Turk":1.0,"Kurd":1.1,"Arab":1.1,"Other":1.1}
    ed_w      = {
        "Illiterate":1.5,"Literate without schooling":1.4,
        "Primary School":1.3,"Middle School":1.2,"High School":1.0,
        "Bachelor’s Degree":0.8,"Master/PhD":0.7
    }
    marital_w = {"Single":1.0,"Married":0.95,"Divorced":1.2}
    emp_w     = {"Employed":0.9,"Unemployed":1.2,"Student":1.0,"Retired":0.7}
    housing_w = {"Houseowner":0.9,"Rent":1.0,"Homeless":2.0}

    # Rastgele veri
    df = pd.DataFrame({
        "age_group": np.random.choice(ages, n_samples, p=age_p),
        "gender":    np.random.choice(genders, n_samples, p=gender_p),
        "race_ethnicity": np.random.choice(races, n_samples, p=race_p),
        "education_level": np.random.choice(ed_lvls, n_samples, p=ed_p),
        "marital_status":  np.random.choice(maritals, n_samples, p=mar_p),
        "employment_status": np.random.choice(emps, n_samples, p=emp_p),
        "housing_status":   np.random.choice(housings, n_samples, p=house_p),
        "has_dependents":   np.random.choice(yesno, n_samples),
        "prior_convictions":      np.random.poisson(3, n_samples),
        "juvenile_convictions":   np.random.poisson(1.5, n_samples),
        "prior_probation_violation": np.random.choice(yesno, n_samples),
        "prior_incarceration":       np.random.choice(yesno, n_samples),
        "substance_abuse_history":   np.random.choice(yesno, n_samples),
        "mental_health_issues":      np.random.choice(yesno, n_samples),
        "gang_affiliation":          np.random.choice(yesno, n_samples),
        "aggression_history":        np.random.choice(yesno, n_samples),
        "compliance_history":        np.random.choice(yesno, n_samples),
        "motivation_to_change":      np.random.choice(yesno, n_samples),
        "stable_employment_past":    np.random.choice(yesno, n_samples),
        "positive_social_support":   np.random.choice(yesno, n_samples)
    })

    # Recidivism olasılığı (log-odds → sigmoid)
    def calc_prob(r):
        s = -2.7
        s += np.log(age_w[r.age_group])
        s += np.log(gender_w[r.gender])
        s += np.log(race_w[r.race_ethnicity])
        s += np.log(ed_w[r.education_level])
        s += np.log(marital_w[r.marital_status])
        s += np.log(emp_w[r.employment_status])
        s += np.log(housing_w[r.housing_status])
        s += 0.25 * r.prior_convictions
        s += 0.35 * r.juvenile_convictions
        s += 0.5  if r.prior_probation_violation else 0
        s += 0.5  if r.prior_incarceration else -0.3
        s += 0.5  if r.substance_abuse_history else -0.3
        s += 0.4  if r.mental_health_issues else 0
        s += 0.8  if r.gang_affiliation else 0
        s += 0.5  if r.aggression_history else -0.3
        s += -0.4 if r.motivation_to_change else 0
        s += 0.5  if not r.compliance_history else 0
        s += -0.3 if r.stable_employment_past else 0.3
        s += -0.3 if r.positive_social_support else 0.3
        s += -0.2 if r.has_dependents else 0.2
        return float(expit(s))

    df["recidivism_prob"] = df.apply(calc_prob, axis=1)

    def assign_cls(p):
        # %5 ihtimalle komşu sınıfa geçiş
        if p < 0.45:
            return np.random.choice([0, 1], p=[0.95, 0.05])
        elif p < 0.7:
            return np.random.choice([1, 0, 2], p=[0.90, 0.05, 0.05])
        else:
            return np.random.choice([2, 1], p=[0.95, 0.05])

    df["recidivism"] = df["recidivism_prob"].apply(assign_cls)
    return df.drop(columns="recidivism_prob")

if __name__ == "__main__":
    df = generate_mock_data(seed=42)
    df.to_csv("mock_data.csv", index=False)
    print("✔ mock_data.csv oluşturuldu.")
    print(df["recidivism"].value_counts(normalize=True))
