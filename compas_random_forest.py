# ---------- RANDOM FOREST SÜRÜMÜ ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
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

# 5) Pipeline: SMOTE ➔ RandomForest
pipeline_rf = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])

# 6) GridSearch hiperparametreleri
param_grid_rf = {
    "clf__n_estimators": [100, 200, 500],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2]
}

grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 7) Eğit
grid_rf.fit(X_train, y_train)
print("RF en iyi parametreler:", grid_rf.best_params_)
print("RF en iyi CV doğruluk:", grid_rf.best_score_)

# 8) Test değerlendirme
best_rf = grid_rf.best_estimator_
y_pred_rf  = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)

print("\nConfusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report (RF):\n", classification_report(y_test, y_pred_rf))
auc_rf = roc_auc_score(pd.get_dummies(y_test), y_proba_rf, multi_class="ovr", average="macro")
print("Test AUC (RF, macro OVR):", auc_rf)

# 9) Class=2 için ROC eğrisi
fpr_rf, tpr_rf, _ = roc_curve((y_test==2).astype(int), y_proba_rf[:,2])
plt.figure(figsize=(7,5))
plt.plot(fpr_rf, tpr_rf, label=f"RF High Risk vs Rest (AUC={roc_auc_score((y_test==2).astype(int), y_proba_rf[:,2]):.2f})")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC - Random Forest (Class=2)")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 10) SHAP açıklayıcılığı
#    (TreeExplainer RF için gayet hızlı)
explainer_rf = shap.TreeExplainer(best_rf.named_steps["clf"])
shap_vals_rf = explainer_rf.shap_values(best_rf.named_steps["smote"].fit_resample(X_train,y_train)[0])
shap.summary_plot(shap_vals_rf, X_test, plot_type="bar", max_display=20)
shap.summary_plot(shap_vals_rf, X_test, max_display=20)
