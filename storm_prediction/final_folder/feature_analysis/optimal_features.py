import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

"""
IL FAUT MODIFIER POUR QUE CA EVALUE UNIQUEMENT LIGHTGBM
Input  : data/data_cleaned.csv
Output : model_output/features_xgb.png + features_lgbm.png + features_selection.png
"""

DATA_PATH  = "../output/data_enrichie_features.csv"
OUTPUT_DIR = "model_output/"
TARGET_COL = "is_last_lightning_cloud_ground"
META_COLS  = ["lightning_id", "airport", "storm_id", "date", "icloud"]


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["icloud"] == False].copy()

    exclude = set(META_COLS + [TARGET_COL])
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and not c.startswith("Q")
                    and not c.endswith("_first")]

    X = df[feature_cols].select_dtypes(include=[np.number])
    # X = X.drop(columns=X.columns[X.isna().mean() > 0.5].tolist())
    # X = X.fillna(X.median())

    y = pd.to_numeric(df.loc[X.index, TARGET_COL], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    return X, y


def plot_importances(name, native_imp, perm_imp, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    native_imp.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title(f"{name} — Importance native (gain)")

    perm_imp.sort_values().plot.barh(ax=axes[1], color="darkorange")
    axes[1].set_title(f"{name} — Permutation Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → {os.path.basename(output_path)}")


def score_top_n(model_cls, model_params, X_train, y_train, X_test, y_test,
                feature_order, ns):
    """Entraîne le modèle sur top-N features et retourne AUC + AP."""
    results = []
    for n in ns:
        feats = feature_order[:n]
        m = model_cls(**model_params)
        m.fit(X_train[feats], y_train)
        proba = m.predict_proba(X_test[feats])[:, 1]
        results.append({
            "n_features": n,
            "roc_auc": roc_auc_score(y_test, proba),
            "avg_precision": average_precision_score(y_test, proba),
        })
    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement...")
    X, y = load(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    xgb_params = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                      eval_metric="aucpr", random_state=42, n_jobs=-1)

    lgbm_params = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
                       random_state=42, n_jobs=-1, verbose=-1)

    ns = [5, 10, 15, 20, X.shape[1]]
    report_lines = ["FEATURES OPTIMALES — no_cg_next_30min", ""]

    for name, model_cls, params in [("XGBoost", XGBClassifier, xgb_params),
                                     ("LightGBM", LGBMClassifier, lgbm_params)]:
        print(f"Entraînement {name}...")
        model = model_cls(**params)
        model.fit(X_train, y_train)

        # Importance native
        native_imp = pd.Series(model.feature_importances_, index=X.columns)
        native_order = native_imp.sort_values(ascending=False).index.tolist()

        # Permutation importance
        # from : scikit-learn.org/stable/modules/permutation_importance.html
        print(f"  Permutation importance {name}...")
        perm = permutation_importance(model, X_test, y_test,
                                      scoring="roc_auc", n_repeats=10,
                                      random_state=42, n_jobs=-1)
        perm_imp = pd.Series(perm.importances_mean, index=X.columns)
        perm_order = perm_imp.sort_values(ascending=False).index.tolist()

        fname = f"features_{name.lower()}.png"
        plot_importances(name, native_imp, perm_imp,
                         os.path.join(OUTPUT_DIR, fname))

        # Sélection progressive (ordre permutation)
        print(f"  Sélection progressive {name}...")
        scores = score_top_n(model_cls, params, X_train, y_train,
                             X_test, y_test, perm_order, ns)

        report_lines += [
            f"--- {name} — Top features (permutation) ---",
            perm_imp.sort_values(ascending=False).head(15).to_string(),
            "",
            f"--- {name} — Score par nb de features ---",
            scores.to_string(index=False),
            "",
        ]

    # Graphique sélection progressive
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in [(axes[0], "roc_auc", "ROC-AUC"),
                               (axes[1], "avg_precision", "Avg Precision")]:
        for name, model_cls, params, color in [
            ("XGBoost", XGBClassifier, xgb_params, "steelblue"),
            ("LightGBM", LGBMClassifier, lgbm_params, "darkorange")
        ]:
            model = model_cls(**params)
            model.fit(X_train, y_train)
            perm = permutation_importance(model, X_test, y_test,
                                          scoring="roc_auc", n_repeats=5,
                                          random_state=42, n_jobs=-1)
            perm_order = pd.Series(perm.importances_mean,
                                   index=X.columns).sort_values(ascending=False).index.tolist()
            scores = score_top_n(model_cls, params, X_train, y_train,
                                 X_test, y_test, perm_order, ns)
            ax.plot(scores["n_features"], scores[metric],
                    marker="o", label=name, color=color)

        ax.set_xlabel("Nb features")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs nb features")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "features_selection.png"), dpi=150)
    plt.close()
    print("  → features_selection.png")

    report = "\n".join(report_lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rapport_features_optimales.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
