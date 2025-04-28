import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Veri yükle
df = pd.read_csv('mock_compas_data_weighted_full.csv')

# 2. Giriş ve çıkışı ayır
X = df.drop(columns=['recidivism'])
y = df['recidivism']

# 3. Kategorik değişkenleri One-Hot Encoding yap
X = pd.get_dummies(X)

# 4. Eğitim ve test setine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeli oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Tahmin yap
y_pred = model.predict(X_test)

# 7. Sonuçları değerlendir
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
