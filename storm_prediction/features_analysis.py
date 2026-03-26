import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

"""
Analyse exploratoire des features de enrichi.csv.
Cible binaire : is_last_lightning_cloud_ground (déjà binaire).
Produit des graphiques dans OUTPUT_DIR.

Input  : data/enrichi.csv
Output : output/feature_analysis/*.png + feature_ranking.csv
"""

FEATURES_PATH = "/home/kraus/databattle2026-meteorage/storm_prediction/src/output/data_enrichie_features.csv"
OUTPUT_DIR    = "/home/kraus/databattle2026-meteorage/storm_prediction/src/output/features_enrichies/"
TOP_N         = 20

META_COLS = [
    "lightning_id", "lightning_airport_id", "date", "lon", "lat",
    "azimuth", "airport", "airport_alert_id", "storm_id",
    "NUM_POSTE", "NOM_USUEL", "AAAAMMJJHH"
]
TARGET_COL = "is_last_lightning_cloud_ground"


def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)

    exclude_cols = set(META_COLS + [TARGET_COL])

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        # Exclure colonnes qualité (Q*) et métadonnées station (*_first)
        if c.startswith("Q") or c.endswith("_first"):
            continue
        feature_cols.append(c)

    X = df[feature_cols].select_dtypes(include=[np.number])

    nan_ratio = X.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
    if cols_to_drop:
        print(f"  {len(cols_to_drop)} colonnes supprimées (>50% NaN)")
    X = X.drop(columns=cols_to_drop)
    X = X.fillna(X.median())

    # from : stackoverflow.com/questions/17605563
    y = df.loc[X.index, TARGET_COL]
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    print(f"  {len(X):,} lignes — {X.shape[1]} features — taux label=1 : {y.mean():.1%}")
    return X, y


def compute_roc_auc_per_feature(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ROC-AUC univarié par feature. Symétrisé autour de 0.5.
    from : scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """
    aucs = {}
    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col])
            # Symétrise : si AUC < 0.5, la feature est informative dans l'autre sens
            aucs[col] = max(auc, 1 - auc)
        except Exception:
            aucs[col] = 0.5
    return pd.Series(aucs, name="roc_auc")


def plot_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    corr = X.corrwith(y).rename("pearson_corr")
    mi = pd.Series(
        mutual_info_classif(X, y, random_state=42),
        index=X.columns,
        name="mutual_info"
    )

    print("  Calcul ROC-AUC par feature...")
    roc_auc = compute_roc_auc_per_feature(X, y)

    ranking = pd.concat([corr.abs().rename("abs_pearson"), mi, roc_auc], axis=1)
    ranking["combined_rank"] = (
        ranking["abs_pearson"].rank(ascending=False) +
        ranking["mutual_info"].rank(ascending=False) +
        ranking["roc_auc"].rank(ascending=False)
    )
    ranking = ranking.sort_values("combined_rank")
    ranking.to_csv(os.path.join(OUTPUT_DIR, "feature_ranking.csv"))

    top = ranking.head(TOP_N)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    colors = ["steelblue" if v >= 0 else "tomato" for v in corr.loc[top.index]]
    axes[0].barh(top.index[::-1], corr.loc[top.index[::-1]], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"Corrélation Pearson (top {TOP_N})")
    axes[0].set_xlabel("Corrélation")

    axes[1].barh(top.index[::-1], mi.loc[top.index[::-1]], color="mediumseagreen")
    axes[1].set_title(f"Mutual Information (top {TOP_N})")
    axes[1].set_xlabel("MI score")

    axes[2].barh(top.index[::-1], roc_auc.loc[top.index[::-1]], color="darkorange")
    axes[2].axvline(0.5, color="black", linewidth=0.8, linestyle="--", label="random")
    axes[2].set_title(f"ROC-AUC univarié (top {TOP_N})")
    axes[2].set_xlabel("AUC")
    axes[2].set_xlim(0.48, None)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_feature_importance.png"), dpi=150)
    plt.close()
    print("  → 1_feature_importance.png")
    return ranking


def plot_correlation_matrix(X: pd.DataFrame, ranking: pd.DataFrame):
    top_features = ranking.head(TOP_N).index.tolist()
    corr_matrix = X[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Matrice de corrélation — top {TOP_N} features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_correlation_matrix.png"), dpi=150)
    plt.close()
    print("  → 2_correlation_matrix.png")


def plot_distributions(X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame):
    top_features = ranking.head(TOP_N).index.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X[top_features]),
        columns=top_features, index=X.index
    )
    y = y.loc[X.index]

    n_cols = 4
    n_rows = int(np.ceil(TOP_N / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        for label, color, name in [(0, "steelblue", "Cloud-to-cloud"),
                                   (1, "tomato",    "Cloud-to-ground")]:
            vals = X_scaled.loc[y == label, feat].dropna()
            ax.hist(vals, bins=40, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribution des top {TOP_N} features par type de foudre (normalisées)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_feature_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 3_feature_distributions.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement...")
    X, y = load(FEATURES_PATH)

    print("Corrélations + Mutual Information")
    ranking = plot_feature_importance(X, y)

    print("Matrice de corrélation inter-features")
    plot_correlation_matrix(X, ranking)

    print("Distributions par label")
    plot_distributions(X, y, ranking)

    print(f"\nTop 10 features (rang combiné) :")
    print(ranking.head(10)[["abs_pearson", "mutual_info", "roc_auc"]].to_string())
    print(f"\nRésultats → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
