import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Veriyi yükle
df = pd.read_csv('mock_compas_data_weighted_full.csv')

# 2. Giriş ve çıkışı ayır
X = df.drop(columns=['recidivism'])
y = df['recidivism']

# 3. Kategorik değişkenleri one-hot encode yap
X = pd.get_dummies(X)

# 4. SMOTE uygulayarak dengele
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Orijinal sınıf dağılımı:\n{y.value_counts()}")
print(f"SMOTE sonrası sınıf dağılımı:\n{pd.Series(y_resampled).value_counts()}")

# 5. Eğitim ve test setine böl
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 6. Modeli kur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Tahmin yap ve değerlendir
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
