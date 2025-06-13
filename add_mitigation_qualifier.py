import pandas as pd
import joblib
import numpy as np

# Veriyi ve modeli yükle
df = pd.read_csv("mock_data_with_probs.csv")
pipeline = joblib.load("recidivism_xgb_pipeline.pkl")

# Boş listelerle başlat
df["mitigations"] = [[] for _ in range(len(df))]
df["qualifiers"] = [[] for _ in range(len(df))]

# Yardımcı fonksiyon
def sample_and_tag(df_subset, percent, label):
    n = int(len(df_subset) * percent)
    sampled_idx = df_subset.sample(n=n, random_state=42).index
    df.loc[sampled_idx, "mitigations"] = df.loc[sampled_idx, "mitigations"].apply(lambda x: x + [label])
    return sampled_idx

# ======================= MITIGATIONS ===========================
# Yaş grubu ve sağır
age_12_14_df = df[df["age_group"] == "12-14"]
df.loc[age_12_14_df.index, "mitigations"] = df.loc[age_12_14_df.index, "mitigations"].apply(lambda x: x + ["age_12-14"])

age_15_17_df = df[df["age_group"] == "15-17"]
df.loc[age_15_17_df.index, "mitigations"] = df.loc[age_15_17_df.index, "mitigations"].apply(lambda x: x + ["age_15-17"])
df.loc[age_15_17_df.sample(frac=0.02, random_state=2).index, "mitigations"] = df.loc[age_15_17_df.sample(frac=0.02, random_state=2).index, "mitigations"].apply(lambda x: x + ["deaf_15-17"])

age_18_24_df = df[df["age_group"] == "18-24"]
df.loc[age_18_24_df.sample(frac=0.02, random_state=3).index, "mitigations"] = df.loc[age_18_24_df.sample(frac=0.02, random_state=3).index, "mitigations"].apply(lambda x: x + ["deaf_18-21"])

# Zihinsel bozukluk
mental_health_df = df[df["mental_health_issues"] == True]
df.loc[mental_health_df.sample(frac=0.3, random_state=4).index, "mitigations"] = df.loc[mental_health_df.sample(frac=0.3, random_state=4).index, "mitigations"].apply(lambda x: x + ["mental_disorder_partially"])

# Unjust provocation
mild = df.sample(frac=0.08, random_state=5).index
df.loc[mild, "mitigations"] = df.loc[mild, "mitigations"].apply(lambda x: x + ["unjust_provocation_mild"])

moderate_pool = df.index.difference(mild)
moderate = df.loc[moderate_pool].sample(frac=0.07, random_state=6).index
df.loc[moderate, "mitigations"] = df.loc[moderate, "mitigations"].apply(lambda x: x + ["unjust_provocation_moderate"])

severe_pool = df.index.difference(mild.union(moderate))
severe = df.loc[severe_pool].sample(frac=0.05, random_state=7).index
df.loc[severe, "mitigations"] = df.loc[severe, "mitigations"].apply(lambda x: x + ["unjust_provocation_severe"])

# ======================= QUALIFIERS ===========================
qualifier_distributions = {
    "premeditated_kill": 0.15,
    "monstrous": 0.06,
    "destructive": 0.02,
    "relative": 0.08,
    "child": 0.04,
    "femicide": 0.10,
    "public_servant": 0.01,
    "to_cover_another_crime": 0.02,
    "failed_crime": 0.01,
    "blood_feud": 0.05,
    "tradition": 0.06
}

total_qualified = int(len(df) * 0.60)
qualified_pool = df.sample(n=total_qualified, random_state=10).index
remaining_pool = qualified_pool.copy()

np.random.seed(11)
for qualifier, ratio in qualifier_distributions.items():
    target = int(ratio * len(df))
    eligible = df.loc[remaining_pool]
    eligible = eligible[(eligible["qualifiers"].apply(len) < 2) & (~eligible["qualifiers"].apply(lambda q: qualifier in q))]
    chosen = eligible.sample(n=min(target, len(eligible)), random_state=np.random.randint(0, 10000)).index
    df.loc[chosen, "qualifiers"] = df.loc[chosen, "qualifiers"].apply(lambda x: x + [qualifier])
    remaining_pool = remaining_pool.difference(chosen)

# ======================= DISCRETIONARY ===========================
sample_and_tag(df[(df["recidivism"] == 0) & (df["qualifiers"].apply(len) == 0)], 0.97, "discretionary_mitigation")
sample_and_tag(df[(df["recidivism"] == 1) & (df["qualifiers"].apply(len) == 0)], 0.51, "discretionary_mitigation")
sample_and_tag(df[(df["recidivism"] == 2) & (df["qualifiers"].apply(len) == 0)], 0.03, "discretionary_mitigation")
sample_and_tag(df[(df["recidivism"] == 0) & (df["qualifiers"].apply(len) > 0)], 0.83, "discretionary_mitigation")
sample_and_tag(df[(df["recidivism"] == 1) & (df["qualifiers"].apply(len) > 0)], 0.27, "discretionary_mitigation")
# rec == 2 and qualifiers dolu → hiçbir discretionary yok

# ==================== ONE-HOT ENCODING ========================
all_mitigations = set(m for sublist in df["mitigations"] for m in sublist)
all_qualifiers = set(q for sublist in df["qualifiers"] for q in sublist)

for m in all_mitigations:
    df[m] = df["mitigations"].apply(lambda x: m in x)

for q in all_qualifiers:
    df[q] = df["qualifiers"].apply(lambda x: q in x)

# ==================== KAYDETME ========================
df.to_csv("mock_data_with_all_features.csv", index=False)
print("✅ mock_data_with_all_features.csv dosyası oluşturuldu.")
