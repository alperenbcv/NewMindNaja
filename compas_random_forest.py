import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression


# 1. Veriyi oku
df = pd.read_csv("mock_compas_data_weighted_full.csv")

# 2. Bağımlı ve bağımsız değişkenleri ayır
X = df.drop(columns=["recidivism"])
y = df["recidivism"]

# 3. One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 4. SMOTE ile veri dengele
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
print(f"\nSMOTE sonrası sınıf dağılımı:\n{pd.Series(y_res).value_counts()}")

# 5. RFE ile en önemli 20 özelliği seç
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
log_reg = LogisticRegression(max_iter=1000)
rfe = RFE(log_reg, n_features_to_select=20)
rfe.fit(X_scaled, y_res)
selected_features = X.columns[rfe.support_]
print(f"\nRFE ile seçilen en önemli 20 özellik:\n{selected_features.tolist()}")

# 6. Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(
    X_res[selected_features], y_res, test_size=0.2, random_state=42
)

# 7. GridSearchCV ile hiperparametre optimizasyonu
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi AUC (CV):", grid_search.best_score_)

# 8. En iyi modeli kullan
best_rf = grid_search.best_estimator_

# 9. Tahminler
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

# 10. Değerlendirme
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))

# 11. ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
