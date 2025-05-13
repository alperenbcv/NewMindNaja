import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.base import clone
import shap
import joblib

# 1) Load data
df = pd.read_csv("mock_data.csv")

# 3) Features / target
X = df.drop(columns="recidivism")
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
pipeline = Pipeline([
    ("pre", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    ))
])

# 7) Hyperparameter grid (simplified for demonstration)
param_grid = {
    "clf__n_estimators": [200],
    "clf__max_depth": [2],
    "clf__learning_rate": [0.1],
    "clf__subsample": [0.8],
    "clf__gamma": [0.1],
    "clf__min_child_weight": [1]
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring="accuracy", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

# 8) Get best model
model = grid.best_estimator_

# 9) Train vs Validation accuracy
print("Train accuracy:     ", accuracy_score(y_train, model.predict(X_train)))
print("Validation accuracy:", accuracy_score(y_test,  model.predict(X_test)))

# 10) Test set evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("AUC (macro OVR):",
      roc_auc_score(pd.get_dummies(y_test), y_proba,
                    multi_class="ovr", average="macro"))

# 11) Feature importances
feat_names = model.named_steps["pre"].get_feature_names_out()
importances = model.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
           .sort_values("importance", ascending=False)
print("\nTop 10 important features:\n", imp_df.head(10))
print("\nLeast 10 important features:\n", imp_df.tail(10))

# 12) Shuffle-label test
y_train_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
shuffle_model = clone(model)
shuffle_model.fit(X_train, y_train_shuffled)
print("\nShuffle-label accuracy:", accuracy_score(y_test, shuffle_model.predict(X_test)))


# 13) SHAP Analysis

# Get fitted transformer and transform data
#fitted_pre = model.named_steps["pre"]
#X_tr = fitted_pre.transform(X_train)  # dense numpy array
#X_te = fitted_pre.transform(X_test)

# Get classifier and create explainer
#clf = model.named_steps["clf"]
#explainer = shap.TreeExplainer(clf)

# Calculate SHAP values - FIXED: Now properly handling multi-class
#shap_values = explainer(X_tr)

# Choose a class for analysis (e.g., class 1)
#class_id = 1

# (a) Global feature importance (bar plot, multi-class)
#shap.summary_plot(
#    shap_values.values,
 #   X_tr,
 #   feature_names=feat_names,
 #   plot_type="bar",
 #   title="Global Feature Importance (mean |SHAP| per class)"
#)

# (b) Beeswarm plot (for single class, training set)
#shap.summary_plot(
#    shap_values.values[:, :, class_id],
#    X_tr,
 #   feature_names=feat_names,
 #   title=f"SHAP Beeswarm Plot for Class {class_id} (Train)"
#)

# (c) Dependence plots: top 3 important features (training set)
#top3 = imp_df["feature"].head(3).tolist()
#for i, feat in enumerate(top3):
#    # Find the column index for the feature
 #   feat_idx = np.where(feat_names == feat)[0][0]
 #   shap.dependence_plot(
 #       feat_idx,
 #       shap_values.values[:, :, class_id],
 #       X_tr,
 #       feature_names=feat_names,
 #       title=f"Dependence of {feat} on Class {class_id}"
 #   )

# (e) Numerical summary: mean absolute SHAP values (training set)
#mabs_tr = np.abs(shap_values.values[:, :, class_id]).mean(axis=0)
#mabs_df_tr = pd.DataFrame({
 #   "feature": feat_names,
 #   "mean_abs_shap": mabs_tr
#}).sort_values("mean_abs_shap", ascending=False)
#print(f"\nClass {class_id} mean absolute SHAP values (Train):\n", mabs_df_tr.head(10))

# (f) Beeswarm plot (for single class, test set)
#shap_values_test = explainer(X_te)
#shap.summary_plot(
 #   shap_values_test.values[:, :, class_id],
 #   X_te,
 #   feature_names=feat_names,
 #   title=f"SHAP Beeswarm Plot for Class {class_id} (Test)"
#)

# 14) Save pipeline to file
joblib.dump(model, "recidivism_xgb_pipeline.pkl")
print("Pipeline saved to recidivism_xgb_pipeline.pkl")

# 15) Manual test cases using trained model
def predict_from_case_dict(model, case_dict):
    df = pd.DataFrame([case_dict])
    probas = model.predict_proba(df)[0]
    predicted_class = np.argmax(probas)
    print(f"\n🎯 Tahmin Sonucu:")
    print("Tahmin edilen sınıf:", predicted_class)
    print("Sınıf olasılıkları: ", {f"class_{i}": f"{100 * p:.2f}%" for i, p in enumerate(probas)})
    print("Girdi:", case_dict)

# Test Case 1 – Genç riskli birey
test_case_1 = {
    "age_group": "12-14",
    "gender": "Male",
    "race_ethnicity": "Turk",
    "education_level": "Primary School",
    "marital_status": "Single",
    "employment_status": "Student",
    "housing_status": "Rent",
    "has_dependents": False,
    "prior_convictions": 0,
    "juvenile_convictions": 1,
    "prior_probation_violation": False,
    "prior_incarceration": False,
    "substance_abuse_history": False,
    "mental_health_issues": False,
    "gang_affiliation": False,
    "aggression_history": False,
    "compliance_history": False,
    "motivation_to_change": False,
    "stable_employment_past": False,
    "positive_social_support": False
}

# Test Case 2 – Düşük riskli yetişkin birey
test_case_2 = {
    "age_group": "45-54",
    "gender": "Female",
    "race_ethnicity": "Turk",
    "education_level": "Master/PhD",
    "marital_status": "Married",
    "employment_status": "Employed",
    "housing_status": "Houseowner",
    "has_dependents": True,
    "prior_convictions": 0,
    "juvenile_convictions": 0,
    "prior_probation_violation": False,
    "prior_incarceration": False,
    "substance_abuse_history": False,
    "mental_health_issues": False,
    "gang_affiliation": False,
    "aggression_history": False,
    "compliance_history": True,
    "motivation_to_change": True,
    "stable_employment_past": True,
    "positive_social_support": True
}

predict_from_case_dict(model, test_case_1)
predict_from_case_dict(model, test_case_2)
