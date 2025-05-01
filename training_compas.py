import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Veriyi oku
df = pd.read_csv("mock_compas_data_weighted_full.csv")

# 2. Bağımlı ve bağımsız değişkenleri ayır
X = df.drop(columns=["recidivism"])
y = df["recidivism"]

# 3. One-hot encoding
X = pd.get_dummies(X)

# 4. SMOTE ile veri dengele
print(f"Orijinal sınıf dağılımı:\n{y.value_counts()}")
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
print(f"\nSMOTE sonrası sınıf dağılımı:\n{pd.Series(y_res).value_counts()}")

# 5. Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 6. XGBoost eğit
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# 7. Feature Importance
importances = xgb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
low_importance = feature_importance_df[feature_importance_df["importance"] < 0.005]
print("\n\u00d6nemsiz (importance \u2248 0) \u00d6zellikler:")
print(low_importance.sort_values(by="importance"))

# 8. Korelasyon matrisi (yüksek korelasyonlu sütunları bul)
corr_matrix = pd.DataFrame(X_res).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper.columns if any(upper[column] > 0.9)]
print(f"\nY\u00fcksek korelasyonlu (corr > 0.9) s\u00fctunlar: {high_corr_features}")

# 9. RFE ile en önemli 20 özelliği seç
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
log_reg = LogisticRegression(max_iter=1000)
rfe = RFE(log_reg, n_features_to_select=20)
rfe.fit(X_scaled, y_res)
selected_features = X.columns[rfe.support_]
print(f"\nRFE ile se\u00e7ilen en \u00f6nemli 20 \u00f6zellik:\n{selected_features.tolist()}")

# 10. Seçilen özelliklerle yeniden eğitim
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_res[selected_features], y_res, test_size=0.2, random_state=42
)
best_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(X_train_sel, y_train_sel)
y_proba = best_model.predict_proba(X_test_sel)[:, 1]

# 11. F1 skoru için en iyi threshold'u bul
precisions, recalls, thresholds = precision_recall_curve(y_test_sel, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"\nF1'e g\u00f6re en iyi threshold: {best_thresh:.2f}")

# 12. Tahmin yap ve değerlendir
y_pred = (y_proba > best_thresh).astype(int)
print(f"\n--- Threshold = {best_thresh:.2f} ---")
print(classification_report(y_test_sel, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_sel, y_pred))

# 13. ROC Curve çiz
fpr, tpr, _ = roc_curve(y_test_sel, y_proba)
auc = roc_auc_score(y_test_sel, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost + RFE + Best Threshold")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
