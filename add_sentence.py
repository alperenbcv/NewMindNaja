import pandas as pd
import joblib
import numpy as np
import random

df = pd.read_csv("mock_data_with_all_features.csv")
pipeline = joblib.load("recidivism_xgb_pipeline.pkl")


def determine_sentence(row):
    base_sentence = "Life Imprisonment"

    has_qualifiers = len(row["qualifiers"]) > 0
    unjust_provocation_mild = row["unjust_provocation_mild"]
    unjust_provocation_moderate = row["unjust_provocation_moderate"]
    unjust_provocation_severe = row["unjust_provocation_severe"]
    age_12_14 = row["age_12-14"]
    age_15_17 = row["age_15-17"]
    deaf_15_17 = row["deaf_15-17"]
    deaf_18_21 = row["deaf_18-21"]
    discretionary_mitigation = row["discretionary_mitigation"]
    recidivism = row["recidivism"]
    mental_disorder_partially = row["mental_disorder_partially"]

    # QUALIFIER → Aggravated Life Imprisonment
    if has_qualifiers:
        base_sentence = "Aggravated Life Imprisonment"

    # UNJUST PROVOCATION
    if unjust_provocation_severe:
        base_sentence = random.randint(216, 288) if base_sentence == "Aggravated Life Imprisonment" else random.randint(
            144, 216)
    elif unjust_provocation_moderate:
        base_sentence = random.randint(234, 270) if base_sentence == "Aggravated Life Imprisonment" else random.randint(
            162, 198)
    elif unjust_provocation_mild:
        base_sentence = random.randint(216, 252) if base_sentence == "Aggravated Life Imprisonment" else random.randint(
            144, 180)

    # AGE 12-14 or DEAF 15-17
    if age_12_14 or deaf_15_17:
        if base_sentence == "Aggravated Life Imprisonment":
            base_sentence = random.randint(144, 180)
        elif base_sentence == "Life Imprisonment":
            base_sentence = random.randint(108, 132)
        elif isinstance(base_sentence, int):
            base_sentence = min(84, base_sentence // 2)

    # AGE 15-17 or DEAF 18-21
    if age_15_17 or deaf_18_21:
        if base_sentence == "Aggravated Life Imprisonment":
            base_sentence = random.randint(216, 288)
        elif base_sentence == "Life Imprisonment":
            base_sentence = random.randint(144, 180)
        elif isinstance(base_sentence, int):
            base_sentence = min(144, int(base_sentence * 2 / 3))

    # PARTIAL MENTAL DISORDER
    if mental_disorder_partially:
        if base_sentence == "Aggravated Life Imprisonment":
            base_sentence = 300
        elif base_sentence == "Life Imprisonment":
            base_sentence = 360
        elif isinstance(base_sentence, int):
            base_sentence = random.randint(int(base_sentence * 5 / 6), int(base_sentence * 11 / 12))

    # DISCRETIONARY MITIGATION
    if discretionary_mitigation:
        if base_sentence == "Aggravated Life Imprisonment":
            base_sentence = "Life Imprisonment"
        elif base_sentence == "Life Imprisonment":
            base_sentence = 300
        elif isinstance(base_sentence, int):
            if recidivism == 0:
                base_sentence = random.randint(int(base_sentence * 5 / 6), int(base_sentence * 8 / 9))
            elif recidivism == 1:
                base_sentence = random.randint(int(base_sentence * 8 / 9), int(base_sentence * 11 / 12))
            elif recidivism == 2:
                base_sentence = random.randint(int(base_sentence * 11 / 12), int(base_sentence * 17 / 18))

    # Final return as Series
    if isinstance(base_sentence, int):
        return pd.Series({
            "sentence_type": "Fixed Term",
            "sentence_amount": base_sentence,
            "is_fixed_term": True
        })
    else:
        return pd.Series({
            "sentence_type": base_sentence,
            "sentence_amount": 0,
            "is_fixed_term": False
        })

# Bu satırı fonksiyonun bitiminden sonra yaz:
df[["sentence_type", "sentence_amount", "is_fixed_term"]] = df.apply(determine_sentence, axis=1)

# İstersen CSV olarak dışa aktarabilirsin:
df.to_csv("mock_data_with_sentences.csv", index=False)