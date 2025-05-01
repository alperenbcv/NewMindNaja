import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Veri setini oku
df = sns.load_dataset("titanic")

# 2. Eksik yaşları rastgele doldurmak için uygun dağılımdan çek
age_values = df['age'].dropna()
df['age'] = df['age'].apply(lambda x: np.random.choice(age_values) if pd.isna(x) else x)

# 3. 'embarked' eksiklerini mod ile doldur
mode_embarked = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(mode_embarked)

# 5. Sadece 'alive' sütununu çıkar
df.drop(columns=['alive'], inplace=True)

# 6. Kategorik sütunları one-hot encode et
df = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'who', 'embark_town', 'deck'], drop_first=True)


# 7. Giriş ve hedef değişkenleri ayır
X = df.drop(columns=['survived'])
y = df['survived']

# 8. Eğitim/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Model kur ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 11. Tahminler
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 12. Değerlendirme
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))

# 13. ROC Curve çiz
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

