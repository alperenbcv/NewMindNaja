# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Veriyi oku
df = pd.read_csv("mock_data.csv")

# One-hot encoding
X = pd.get_dummies(df.drop(columns="recidivism"), drop_first=True)
y = df["recidivism"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(random_state=42, early_stopping=True))
])

# Parametre arama
param_grid = {
    "clf__hidden_layer_sizes": [(50, 50), (100,), (200,)],
    "clf__activation": ["tanh", "relu"],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [1e-3, 1e-2],
    "clf__max_iter": [1000]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X_train_bal, y_train_bal)

# En iyi model ve sonuç
best_model = grid.best_estimator_
print(">> En iyi parametreler:", grid.best_params_)

def eval_model(m, Xv, yv, name):
    y_pred = m.predict(Xv)
    y_pr = m.predict_proba(Xv)
    print(f"\n--- {name} classification report ---")
    print(classification_report(yv, y_pred))
    print(f"AUC (macro OVR): {roc_auc_score(pd.get_dummies(yv), y_pr, multi_class='ovr', average='macro'):.3f}")

eval_model(best_model, X_test, y_test, "Test Set")

import joblib

# Modeli diske kaydet
joblib.dump(best_model, "recidivism_model.pkl")
print("✔ Model recidivism_model.pkl dosyasına kaydedildi.")

