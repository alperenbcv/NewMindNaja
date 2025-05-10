import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# mock_data_compas.py içindeki generate_mock_data fonksiyonunu kullandığınız varsayılıyor
from mock_data_compas import generate_mock_data

def prepare_data():
    df = pd.read_csv("mock_data.csv")
    X = df.drop(columns="recidivism")
    y = df["recidivism"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    categorical_cols = [
        "age_group", "gender", "race_ethnicity", "education_level",
        "marital_status", "employment_status", "housing_status"
    ]
    numeric_cols = [
        "has_dependents", "prior_convictions", "juvenile_convictions",
        "prior_probation_violation", "prior_incarceration",
        "substance_abuse_history", "mental_health_issues",
        "gang_affiliation", "aggression_history", "compliance_history",
        "motivation_to_change", "stable_employment_past",
        "positive_social_support"
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    return X_train, X_test, y_train, y_test, preprocessor

def get_models(preprocessor):
    models = {
        "mlp": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", MLPClassifier(random_state=42, early_stopping=True))
            ]),
            {
                "clf__hidden_layer_sizes": [(50, 50), (100,)],
                "clf__activation": ["relu"],
                "clf__alpha": [0.01],
                "clf__learning_rate_init": [0.001],
                "clf__max_iter": [500]
            }
        ),
        "random_forest": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            {
                "clf__n_estimators": [100],
                "clf__max_depth": [None, 10]
            }
        ),
        "logistic_regression": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42, multi_class="ovr"))
            ]),
            {
                "clf__C": [1.0]
            }
        ),
        "decision_tree": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", DecisionTreeClassifier(random_state=42))
            ]),
            {
                "clf__max_depth": [None, 10]
            }
        ),
        "catboost": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", CatBoostClassifier(verbose=0, random_state=42))
            ]),
            {
                "clf__iterations": [100],
                "clf__learning_rate": [0.1],
                "clf__depth": [6]
            }
        ),
        "xgboost": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
            ]),
            {
                "clf__n_estimators": [100],
                "clf__max_depth": [6],
                "clf__learning_rate": [0.1]
            }
        ),
        "lightgbm": (
            Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", LGBMClassifier(random_state=42))
            ]),
            {
                "clf__n_estimators": [100],
                "clf__learning_rate": [0.1],
                "clf__num_leaves": [31]
            }
        )
    }
    return models

def train_and_evaluate_all():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    models = get_models(preprocessor)

    for name, (pipeline, params) in models.items():
        print(f"\n🔧 Training model: {name}")
        grid = GridSearchCV(pipeline, params, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"✅ Best params for {name}: {grid.best_params_}")

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        print(f"\n--- {name.upper()} Classification Report ---")
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class='ovr', average='macro')
        print(f"AUC (macro OVR): {auc:.3f}")

        joblib.dump(best_model, f"{name}_recidivism_model.pkl")
        print(f"💾 Model kaydedildi: {name}_recidivism_model.pkl")

if __name__ == "__main__":
    train_and_evaluate_all()
