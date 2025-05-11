import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import category_encoders as ce
from sklearn.base import clone


def add_custom_features(X):
    """Feature engineering"""
    X = X.copy()
    # Sayısal özellikler üzerinde işlemler
    X['prior_to_juvenile_ratio'] = X['prior_convictions'] / (X['juvenile_convictions'] + 1e-6)
    X['stability_score'] = (X['stable_employment_past'].astype(int) +
                            X['positive_social_support'].astype(int) -
                            X['prior_incarceration'].astype(int) )
    return X


def safe_target_encoding(X, y, categorical_cols):
    """Data leakage önleyerek target encoding"""
    encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=20.0)
    X_encoded = X.copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        encoder.fit(X_train_fold[categorical_cols], y_train_fold)
        X_encoded.loc[val_idx, categorical_cols] = encoder.transform(X_val_fold[categorical_cols])

    return X_encoded, encoder


def preprocess_data(X, y, categorical_cols, scaler=None, fit_scaler=False):
    """Veri ön işleme adımları"""
    # 1) Özellik mühendisliği
    X = add_custom_features(X)

    # 2) Target encoding
    X_encoded, encoder = safe_target_encoding(X, y, categorical_cols)

    # 3) Sadece sayısal sütunları seç
    numeric_cols = X_encoded.select_dtypes(include=['number']).columns
    X_numeric = X_encoded[numeric_cols]

    # 4) Ölçekleme
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
    else:
        X_scaled = scaler.transform(X_numeric)

    return X_scaled, encoder, scaler


def train_and_evaluate():
    # Veri yükleme
    df = pd.read_csv("mock_data.csv")
    X = df.drop(columns="recidivism")
    y = df["recidivism"]

    # Kategorik sütunlar
    categorical_cols = [
        "age_group", "gender", "race_ethnicity", "education_level",
        "marital_status", "employment_status", "housing_status"
    ]

    # Ön işleme (scaler fit ederek)
    X_processed, encoder, scaler = preprocess_data(X, y, categorical_cols, fit_scaler=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, stratify=y, test_size=0.2, random_state=42
    )

    # Model tanımı
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        colsample_bytree=0.6,
        gamma=0.1,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        n_estimators=100,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.6
    )

    # Model eğitimi
    model.fit(X_train, y_train)

    # Değerlendirme
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"AUC (macro OVR): {roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class='ovr', average='macro'):.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()

    # Shuffle test
    def robust_shuffle_test(X, y, model, n_iter=5):
        scores = []
        for _ in range(n_iter):
            y_shuffled = y.sample(frac=1.0)
            temp_model = clone(model)
            temp_model.fit(X, y_shuffled)
            scores.append(temp_model.score(X_test, y_test))
        return np.mean(scores)

    shuffle_score = robust_shuffle_test(X_train, y_train, model)
    print(f"\nRobust shuffle-test score: {shuffle_score:.3f} (should be ~0.33)")

    # Model kaydetme
    joblib.dump({
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "categorical_cols": categorical_cols
    }, "recidivism_model.pkl")
    print("✔ Model saved as recidivism_model.pkl")


if __name__ == "__main__":
    train_and_evaluate()