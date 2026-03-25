import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)

"""
Entraîne un LightGBM sur data_cleaned.csv (filtre icloud=False).
Cible : is_last_lightning_cloud_ground.
Gère le déséquilibre via is_unbalance=True.

Input  : data/data_cleaned.csv
Output : model_output/rapport_lgbm.txt
"""

DATA_PATH  = "../output/data_cleaned.csv"
OUTPUT_DIR = "model_output/"
TARGET_COL = "is_last_lightning_cloud_ground"
META_COLS  = ["lightning_id", "airport", "storm_id", "date", "icloud"]


def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["icloud"] == False].copy()

    exclude = set(META_COLS + [TARGET_COL])
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and not c.startswith("Q")
                    and not c.endswith("_first")]

    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.drop(columns=X.columns[X.isna().mean() > 0.5].tolist())
    X = X.fillna(X.median())

    y = pd.to_numeric(df.loc[X.index, TARGET_COL], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)

    print(f"  {len(X):,} lignes — {X.shape[1]} features — taux label=1 : {y.mean():.1%}")
    return X, y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement...")
    X, y = load(DATA_PATH)

    print("Split train/test (80/20 stratifié)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement LightGBM...")
    # from : lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        metric="average_precision",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    print("Évaluation...")
    y_pred  = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)[:, 1]

    roc_auc       = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    importances = pd.Series(lgbm.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)

    report_lines = [
        "=" * 60,
        "LIGHTGBM — is_last_lightning_cloud_ground (icloud=False)",
        "=" * 60,
        f"Lignes train : {len(X_train):,}  |  test : {len(X_test):,}",
        f"Features     : {X.shape[1]}",
        f"Taux label=1 : {y.mean():.1%}",
        "",
        f"ROC-AUC (test)            : {roc_auc:.4f}",
        f"Average Precision (test)  : {avg_precision:.4f}",
        "",
        "--- Classification Report (seuil 0.5) ---",
        classification_report(y_test, y_pred,
                              target_names=["cloud-to-cloud", "cloud-to-ground"]),
        "",
        "--- Feature Importances (top 15) ---",
    ]
    for feat, imp in importances.items():
        report_lines.append(f"  {feat:<20} {imp}")

    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "rapport_lgbm.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n→ {report_path}")


if __name__ == "__main__":
    main()
