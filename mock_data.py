import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1) SENTETİK VERİ ÜRETİMİ
# -------------------------------
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

# Veri üretimi
data = generate_mock_data(seed=42)
holdout = generate_mock_data(seed=1234, n_samples=1000)

# One-hot encoding
X = pd.get_dummies(data.drop(columns="recidivism"), drop_first=True)
y = data["recidivism"]
X_hold = pd.get_dummies(holdout.drop(columns="recidivism"), drop_first=True)
y_hold = holdout["recidivism"]
X_hold = X_hold.reindex(columns=X.columns, fill_value=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# SMOTE uygulanıyor
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Pipeline (SMOTE yok çünkü zaten uygulandı)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(random_state=42, early_stopping=True))
])

param_grid = {
    "clf__hidden_layer_sizes": [(50, 50), (100,), (200,)],
    "clf__activation": ["tanh", "relu"],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [1e-3, 1e-2],
    "clf__max_iter": [1000]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X_train_bal, y_train_bal)

print(">> En iyi parametreler:", grid.best_params_)
best_model = grid.best_estimator_

# Değerlendirme fonksiyonu
def eval_model(m, Xv, yv, name):
    y_pred = m.predict(Xv)
    y_pr = m.predict_proba(Xv)
    print(f"\n--- {name} classification report ---")
    print(classification_report(yv, y_pred))
    print(f"AUC (macro OVR): {roc_auc_score(pd.get_dummies(yv), y_pr, multi_class='ovr', average='macro'):.3f}")

eval_model(best_model, X_test, y_test, "Test Set")
eval_model(best_model, X_hold, y_hold, "Hold-Out Set")

# -------------------------------
# 6) SHUFFLE ETİKET DENEYİ (memorization kontrolü)
# -------------------------------
# Eğitim etiketlerini karıştıralım
y_train_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
X_train_shuf     = X_train.reset_index(drop=True)

# best_model'ı klonlayıp shuffle edilmiş etiketle yeniden eğit
pipe_shuf = clone(best_model)
pipe_shuf.fit(X_train_shuf, y_train_shuffled)

eval_model(pipe_shuf, X_test, y_test, "Test (Shuffle Etiket)")
eval_model(pipe_shuf, X_hold, y_hold, "Hold-Out (Shuffle Etiket)")

# -------------------------------
# 7) LEARNING CURVE
# -------------------------------
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=np.linspace(0.1,1.0,10),
    cv=5, scoring="accuracy", n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
val_mean   = np.mean(val_scores,   axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label="Eğitim Skoru")
plt.plot(train_sizes, val_mean,   label="Validasyon Skoru")
plt.xlabel("Eğitim Seti Boyutu")
plt.ylabel("Doğruluk")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve

def plot_roc(y_true, y_proba, class_id):
    fpr, tpr, _ = roc_curve((y_true==class_id).astype(int), y_proba[:,class_id])
    plt.plot(fpr, tpr, label=f"Class {class_id} (AUC={roc_auc_score((y_true==class_id).astype(int), y_proba[:,class_id]):.2f})")

plt.figure(figsize=(7,5))
for i in [0,1,2]: plot_roc(y_test, best_model.predict_proba(X_test), i)
plt.plot([0,1],[0,1],"--",color="gray")
plt.legend(); plt.title("ROC Curves per Class"); plt.grid(); plt.show()