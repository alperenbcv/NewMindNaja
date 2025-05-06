import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import matplotlib.pyplot as plt
import joblib

# mock_data_compas.py içindeki veri üreticiyi içe aktar
from mock_data_compas import generate_mock_data

def train_and_evaluate():
    # 1) Veri yükle ve böl
    df = pd.read_csv("mock_data.csv")
    X = df.drop(columns="recidivism")
    y = df["recidivism"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 2) Preprocessing tanımı
    categorical_cols = [
        "age_group","gender","race_ethnicity","education_level",
        "marital_status","employment_status","housing_status"
    ]
    numeric_cols = [
        "has_dependents","prior_convictions","juvenile_convictions",
        "prior_probation_violation","prior_incarceration",
        "substance_abuse_history","mental_health_issues",
        "gang_affiliation","aggression_history","compliance_history",
        "motivation_to_change","stable_employment_past",
        "positive_social_support"
    ]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    # 3) Pipeline: Preprocessing → SMOTE → LogisticRegression
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 4) Hiperparametre arama
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=5,
        scoring="accuracy", verbose=1,
        n_jobs=-1, return_train_score=True
    )
    grid.fit(X_train, y_train)
    print(">> En iyi parametreler:", grid.best_params_)

    best_model = grid.best_estimator_

    # 5) Test set değerlendirme
    y_pred = best_model.predict(X_test)
    y_pr = best_model.predict_proba(X_test)
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(
        pd.get_dummies(y_test), y_pr,
        multi_class="ovr", average="macro"
    )
    print(f"AUC (macro OVR): {auc:.3f}")

    # 6) Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1,1,5), n_jobs=-1
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()

    # 7) Permutation Importance (özellik katkıları)
    # Ön işlemciden feature isimlerini al
    feat_names = best_model.named_steps["pre"].get_feature_names_out()

    # X_test'i preprocess et
    X_test_transformed = best_model.named_steps["pre"].transform(X_test)

    # SMOTE uygulanmadığı için orijinal clf'yi kullan
    clf = best_model.named_steps["clf"]

    # Permutation importance'ı sadece clf ve preprocess edilmiş X ile hesapla
    r = permutation_importance(
        clf, X_test_transformed, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    # DataFrame ile göster
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": r.importances_mean
    }).sort_values("importance", ascending=False)

    print("\n✔ Permutation Importance (Top 20):")
    print(imp_df.head(20))

    # Özellik isimlerini doğru şekilde al
    feat_names = best_model.named_steps["pre"].get_feature_names_out()

    # Uzunlukları karşılaştır, eşleşiyorsa dataframe oluştur
    if len(feat_names) == len(r.importances_mean):
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": r.importances_mean
        }).sort_values("importance", ascending=False)
        print("\nPermutation Importance (Top 20):")
        print(imp_df.head(20))
    else:
        print("❌ Özellik ismi sayısı ile önem skorları eşleşmiyor!")
        print(f"Feature names: {len(feat_names)}, Importances: {len(r.importances_mean)}")

    # 8) Yeni seed ile test
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

    # 12) Logistic Regression ağırlıkları (koefisiyentler)
    clf = best_model.named_steps["clf"]
    feat_names = best_model.named_steps["pre"].get_feature_names_out()

    # Koefisiyentleri al (her sınıf için ayrı)
    coef_df = pd.DataFrame(clf.coef_, columns=feat_names)
    coef_df["class"] = clf.classes_  # Her satır bir sınıf için

    # Uzun formatta göster (özellik, sınıf, ağırlık)
    coef_long = coef_df.melt(id_vars="class", var_name="feature", value_name="weight")

    # En düşük ağırlıklı (önemsiz) özellikleri göster
    print("\n🔍 Önem verilmeyen özellikler (ağırlığa göre):")
    print(coef_long.sort_values(by="weight", key=abs).head(20))

    # 10) Shuffle-label kontrolü
    y_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffle_pipe = clone(best_model)
    shuffle_pipe.fit(X_train, y_shuffled)
    print(f"\nShuffle-label test accuracy: {shuffle_pipe.score(X_test, y_test):.3f}")

    # 11) Kaydet
    joblib.dump(best_model, "recidivism_logreg_pipeline.pkl")
    print("✔ Logistic Regression pipeline kaydedildi: recidivism_logreg_pipeline.pkl")

if __name__ == "__main__":
    train_and_evaluate()
