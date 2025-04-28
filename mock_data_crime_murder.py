import pandas as pd
import numpy as np

# -------------------------------
# Kasten Öldürme Mock Data Üretimi (İndirim Sebepleri Dahil)
# -------------------------------
np.random.seed(44)
n_samples = 300

# Suç Tipi Sabit: Kasten Öldürme (Murder)
crime_type = "Murder"

# Nitelikli Haller
qualified_conditions = [
    "planned",  # Tasarlayarak
    "cruelty",  # Canavarca hisle
    "dangerous_method",  # Yangın, bombalama vb.
    "close_relative",  # Üstsoy, altsoy, eş, kardeş
    "child_or_defenseless",  # Çocuk veya savunmasız kişi
    "pregnant_woman",  # Gebe kadına karşı
    "public_officer",  # Kamu görevi nedeniyle
    "concealment_or_evidence",  # Suçu gizlemek/delil yok etmek
    "blood_feud",  # Kan gütme saikiyle
    "custom"  # Töre saikiyle
]

# İndirim Sebepleri
reduction_reasons = [
    "self_defense",
    "necessity",
    "full_mental_disorder",
    "partial_mental_disorder",
    "provocation",
    "deaf_mute",
    "alcohol_or_drugs_accidental"
]

# Yaş grupları
age_groups = ["under_12", "12_15", "15_18", "18_21", "adult"]

data = []

for _ in range(n_samples):
    record = {"crime_type": crime_type}

    # Her nitelikli hal rastgele True/False
    for cond in qualified_conditions:
        record[cond] = np.random.choice([True, False], p=[0.2, 0.8])

    # İndirim sebepleri rastgele True/False
    for reason in reduction_reasons:
        record[reason] = np.random.choice([True, False], p=[0.05, 0.95])

    # Yaş grubu seçimi
    record["age_group"] = np.random.choice(age_groups, p=[0.02, 0.08, 0.15, 0.15, 0.6])

    # Ceza hesaplama
    base_sentence = 11976  # Müebbet (998 yıl)
    qualified_true_count = sum(record[cond] for cond in qualified_conditions)
    if qualified_true_count > 0:
        base_sentence = 11988  # Ağırlaştırılmış Müebbet (999 yıl)

    final_sentence = base_sentence

    # Önce cezai sorumluluğu tamamen kaldıran nedenler kontrol edilir
    if record["self_defense"] or record["necessity"] or record["full_mental_disorder"] or record["age_group"] == "under_12" or (record["deaf_mute"] and record["age_group"] == "12_15"):
        final_sentence = 0
    else:
        # İndirim uygulamaları
        if record["provocation"]:
            if base_sentence == 11988:
                final_sentence = np.random.randint(216, 289)  # 18-24 yıl
            else:
                final_sentence = np.random.randint(144, 217)  # 12-18 yıl
        elif record["partial_mental_disorder"]:
            if base_sentence == 11988:
                final_sentence = 300  # 25 yıl
            else:
                final_sentence = 240  # 20 yıl
        elif record["age_group"] == "12_15":
            if base_sentence == 11988:
                final_sentence = np.random.randint(144, 181)  # 12-15 yıl
            else:
                final_sentence = np.random.randint(108, 133)  # 9-11 yıl
        elif record["age_group"] == "15_18":
            if base_sentence == 11988:
                final_sentence = np.random.randint(216, 289)  # 18-24 yıl
            else:
                final_sentence = np.random.randint(144, 181)  # 12-15 yıl
        elif record["deaf_mute"] and record["age_group"] == "15_18":
            if base_sentence == 11988:
                final_sentence = np.random.randint(144, 181)  # 12-15 yıl
            else:
                final_sentence = np.random.randint(108, 133)  # 9-11 yıl
        elif record["deaf_mute"] and record["age_group"] == "18_21":
            if base_sentence == 11988:
                final_sentence = np.random.randint(216, 289)  # 18-24 yıl
            else:
                final_sentence = np.random.randint(144, 181)  # 12-15 yıl
        elif record["alcohol_or_drugs_accidental"]:
            final_sentence = 0

    record["sentence_months"] = final_sentence

    data.append(record)

# DataFrame oluştur
murder_df = pd.DataFrame(data)

# CSV Kaydet
murder_df.to_csv("mock_murder_data.csv", index=False)

print("Kasten Öldürme suçu için indirim sebepleri entegre edilerek mock veri başarıyla oluşturuldu ve 'mock_murder_data.csv' dosyasına kaydedildi!")
