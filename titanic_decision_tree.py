import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Titanic veri setini yükle
df = sns.load_dataset("titanic")

# 2. Eksik yaşları rastgele benzer dağılımdan doldur
age_values = df['age'].dropna()
df['age'] = df['age'].apply(lambda x: np.random.choice(age_values) if pd.isna(x) else x)

# 3. 'embarked' eksiklerini mod ile doldur
mode_embarked = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(mode_embarked)

# 4. 'alive' sütununu çıkar (hedef değişkenle aynı anlamda)
df.drop(columns=['alive'], inplace=True)

# 5. Kategorik sütunları one-hot encode et (multicollinearity riskini azaltmak için drop_first=True)
df = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'who', 'embark_town', 'deck'], drop_first=True)

# 6. Giriş ve hedef değişkenleri ayır
X = df.drop(columns=['survived'])
y = df['survived']

# 7. Eğitim/test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. GridSearchCV ile en iyi max_depth değerini bul
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, None]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("En iyi max_depth:", grid_search.best_params_['max_depth'])
print("En iyi doğruluk (CV):", grid_search.best_score_)

# 9. En iyi modeli al
best_tree_model = grid_search.best_estimator_

# 10. Tahmin ve değerlendirme
y_pred = best_tree_model.predict(X_test)
y_proba = best_tree_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))

# 11. ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 12. Karar Ağacını Görselleştir
plt.figure(figsize=(20, 10))
plot_tree(best_tree_model, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
plt.title(f"Karar Ağacı (max_depth = {best_tree_model.get_params()['max_depth']})")
plt.show()
