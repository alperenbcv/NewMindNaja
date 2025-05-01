# ---------- XGBOOST SÜRÜMÜ ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap

# 1) Veriyi oku
data = pd.read_csv("mock_compas_data_weighted_full.csv")
print("Risk level dağılımı:\n", data["recidivism"].value_counts(), "\n")

# 2) Özellikler / hedef
X = data.drop(columns=["recidivism"])
y = data["recidivism"]

# 3) One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 4) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5) Pipeline: SMOTE ➔ XGBClassifier
pipeline_xgb = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    ))
])

# 6) GridSearch hiperparametreleri
param_grid_xgb = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [4, 6, 8],
    "clf__learning_rate": [0.01, 0.1],
    "clf__subsample": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=pipeline_xgb,
    param_grid=param_grid_xgb,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 7) Eğit
grid_xgb.fit(X_train, y_train)
print("XGB en iyi parametreler:", grid_xgb.best_params_)
print("XGB en iyi CV doğruluk:", grid_xgb.best_score_)

# 8) Test değerlendirme
best_xgb = grid_xgb.best_estimator_
y_pred_xgb  = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)

print("\nConfusion Matrix (XGB):\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report (XGB):\n", classification_report(y_test, y_pred_xgb))
auc_xgb = roc_auc_score(pd.get_dummies(y_test), y_proba_xgb, multi_class="ovr", average="macro")
print("Test AUC (XGB, macro OVR):", auc_xgb)


