import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from mock_data_compas import generate_mock_data

def train_and_evaluate():
    # 1) Load data
    df = pd.read_csv("mock_data.csv")
    X = df.drop(columns="recidivism")
    y = df["recidivism"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 2) Preprocessing
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

    # 3) Pipeline
    pipe = Pipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4) Hyperparameter tuning
    param_grid = {
        "clf__n_estimators": [200],
        "clf__max_depth": [6],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [1]
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=3,
        scoring="accuracy", verbose=1,
        n_jobs=-1, return_train_score=True
    )
    grid.fit(X_train, y_train)
    print(">> En iyi parametreler:", grid.best_params_)

    best_model = grid.best_estimator_

    # 5) Evaluate on test
    y_pred = best_model.predict(X_test)
    y_pr = best_model.predict_proba(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(pd.get_dummies(y_test), y_pr, multi_class="ovr", average="macro")
    print(f"AUC (macro OVR): {auc:.3f}")

    # 6) Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=3, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()

    # 7) Permutation importance
    feat_names = best_model.named_steps["pre"].get_feature_names_out()
    X_test_proc = best_model.named_steps["pre"].transform(X_test)
    clf = best_model.named_steps["clf"]
    r = permutation_importance(clf, X_test_proc, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": r.importances_mean
    }).sort_values("importance", ascending=False)
    print("\nPermutation Importance (Top 20):")
    print(imp_df.head(20))

    # 8) New-seed test
    df_new = generate_mock_data(seed=99, n_samples=3000)
    X_new = df_new.drop(columns="recidivism")
    y_new = df_new["recidivism"]
    print(f"\nNew-seed test accuracy: {best_model.score(X_new, y_new):.3f}")

    # 9) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()

    # 10) Shuffle-label test (with aligned index)
    rng = np.random.default_rng(42)
    y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    shuffle_pipe = clone(best_model)
    shuffle_pipe.fit(X_train, y_shuffled)
    shuffle_acc = shuffle_pipe.score(X_test, y_test)
    print(f"\nShuffle-label test accuracy: {shuffle_acc:.3f} (should be close to 0.33 if no signal)")

    # 11) Save model
    joblib.dump(best_model, "recidivism_rf_smote_balanced.pkl")
    print("✔ Model saved as recidivism_rf_smote_balanced.pkl")

if __name__ == "__main__":
    train_and_evaluate()
