import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 1) Load data
df = pd.read_csv("mock_data_with_probs.csv")

# 3) Features / target
cols_to_drop = [c for c in ["recidivism", "recidivism_pred", "prediction_probability"] if c in df.columns]
X = df.drop(columns=cols_to_drop)
y = df["recidivism"]

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5) Preprocessing definition (dense output OneHotEncoder)
cat_cols = [
    "age_group", "gender", "race_ethnicity", "education_level",
    "marital_status", "employment_status", "housing_status"
]
num_cols = [
    "has_dependents", "prior_convictions", "juvenile_convictions",
    "prior_probation_violation", "prior_incarceration",
    "substance_abuse_history", "mental_health_issues",
    "gang_affiliation", "aggression_history", "compliance_history",
    "motivation_to_change", "stable_employment_past", "positive_social_support"
]
preprocessor = ColumnTransformer([
    ("cat",
     OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
     cat_cols),
    ("num", StandardScaler(), num_cols),
])

# 6) Pipeline (SMOTE + XGBoost)
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_estimators=200,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    gamma=0.1,
    min_child_weight=1,
    random_state=42
)

calibrated_xgb = CalibratedClassifierCV(
    estimator=xgb,
    method='isotonic',
    cv=5
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", calibrated_xgb)
])

pipeline.fit(X_train, y_train)

# 9) Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC (macro):", roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class="ovr", average="macro"))

# 10) Save model
joblib.dump(pipeline, "recidivism_xgb_pipeline.pkl")
print("✅ Model saved as recidivism_xgb_pipeline.pkl")

