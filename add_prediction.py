import pandas as pd
import joblib
import numpy as np

# 1) Veriyi ve modeli yükle
df = pd.read_csv("mock_data_with_probs.csv")
pipeline = joblib.load("recidivism_xgb_pipeline.pkl")

# 2) Özellikleri ayır
X = df.drop(columns="recidivism")

# 3) Tahmin ve olasılıkları al
y_pred = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)

# 4) En yüksek olasılığı ve karşılık gelen sınıfı bul
predicted_probs = np.max(y_proba, axis=1) * 100  # yüzde formatında çevir
predicted_class = np.argmax(y_proba, axis=1)

# 5) Yeni sütunları ekle
df["recidivism_pred"] = predicted_class
df["prediction_probability"] = predicted_probs.round(2)

# 6) Dosyayı güncelle
df.to_csv("mock_data_with_probs.csv", index=False)
print("✅ mock_data_with_probs.csv başarıyla kaydedildi.")
