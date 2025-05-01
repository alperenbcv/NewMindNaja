import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import shap

# 1. Veriyi oku
df = pd.read_csv("mock_compas_data_weighted_full.csv")

# 2. Özellikler ve hedef değişkeni ayır
X = df.drop(columns=["recidivism"])
y = df["recidivism"]

# 3. Kategorikleri one-hot encoding ile dönüştür
X = pd.get_dummies(X, drop_first=True)

# 4. SMOTE ile dengele (random_state ekleyerek tekrarlanabilirlik)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("SMOTE sonrası sınıf dağılımı:\n", pd.Series(y_res).value_counts(), "\n")

# 5. Eğitim/test seti (stratify ile orantıyı koru)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)

# 6. Özellikleri ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Feature isimlerini sakla
feature_names = X.columns

# 7. Hiperparametre araması
param_grid = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.001],
    'learning_rate_init': [0.01],
    'max_iter': [2000]
}
mlp = MLPClassifier(random_state=42, early_stopping=True)
grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_scaled, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi CV AUC:", grid.best_score_)

# 8. En iyi modeli değerlendir
best_mlp = grid.best_estimator_
y_pred  = best_mlp.predict(X_test_scaled)
y_proba = best_mlp.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, y_proba))

# 9. ROC eğrisini çiz
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MLPClassifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. SHAP ile açıklayıcı analiz
#    - Ölçeklenmiş dizileri DataFrame'e çevir
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df  = pd.DataFrame(X_test_scaled,  columns=feature_names)

#    - SHAP arka planı için rastgele 100 örnek al
background = X_train_df.sample(n=100, random_state=42)

#    - Sadece "recidivism=1" olasılığını verecek fonksiyon
f_proba1 = lambda x: best_mlp.predict_proba(x)[:, 1]

explainer = shap.KernelExplainer(f_proba1, background, link="logit")
#    - nsamples parametresi ile hesaplamayı hızlandırabiliriz
shap_values = explainer.shap_values(X_test_df, nsamples=100)

#    - Özet grafiği
shap.summary_plot(shap_values, X_test_df)
