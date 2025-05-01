import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap

# 1) Veriyi oku ve dağılımı kontrol et
data = pd.read_csv("mock_compas_data_weighted_full.csv")
print("Risk level dağılımı:\n", data["recidivism"].value_counts(), "\n")

# 2) Özellikler / hedef
X = data.drop(columns=["recidivism"])
y = data["recidivism"]

# 3) One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 4) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5) Pipeline: SMOTE ➔ Ölçekleme ➔ MLPClassifier
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(random_state=42, early_stopping=True))
])

# 6) GridSearch hiperparametreleri
param_grid = {
    "clf__hidden_layer_sizes": [(50, 50), (100,), (200,)],
    "clf__activation": ["tanh", "relu"],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [1e-3, 1e-2],
    "clf__max_iter": [1000, 2000]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 7) Eğit
grid.fit(X_train, y_train)
print("En iyi parametreler:", grid.best_params_)
print("En iyi CV doğruluk:", grid.best_score_)

# 8) Test değerlendirme
best = grid.best_estimator_
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9) ROC AUC (macro, OVR)
auc_macro = roc_auc_score(pd.get_dummies(y_test), y_proba, average="macro", multi_class="ovr")
print("Test AUC (macro OVR):", auc_macro)

# 10) High risk (class=2) için ROC eğrisi
fpr, tpr, _ = roc_curve((y_test == 2).astype(int), y_proba[:, 2])
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"High Risk vs Rest (AUC = {roc_auc_score((y_test == 2).astype(int), y_proba[:, 2]):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - High Risk vs Others")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 11) SHAP açıklayıcılığı (sınıf 2 - yüksek risk)
feature_names = X.columns
X_tr_resampled, _ = best.named_steps["smote"].fit_resample(X_train, y_train)
X_tr_scaled = best.named_steps["scaler"].transform(X_tr_resampled)
X_te_scaled = best.named_steps["scaler"].transform(X_test)

X_tr_df = pd.DataFrame(X_tr_scaled, columns=feature_names)
X_te_df = pd.DataFrame(X_te_scaled, columns=feature_names)

background = X_tr_df.sample(100, random_state=42)
f_high = lambda x: best.named_steps["clf"].predict_proba(x)[:, 2]

explainer = shap.KernelExplainer(f_high, background, link="logit")
shap_values = explainer.shap_values(X_te_df, nsamples=100)

# SHAP summary plot
shap.summary_plot(shap_values, X_te_df)
