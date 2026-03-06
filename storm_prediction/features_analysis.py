import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

"""
Analyse exploratoire des features de storms.csv.
Crée un label binaire à partir de n_lightnings (médiane)
et analyse les features météo agrégées.

Produit des graphiques dans OUTPUT_DIR.
Input  : data/storms.csv
Output : output/feature_analysis/*.png + feature_ranking.csv
"""

FEATURES_PATH = "/home/dev/databattle2026-meteorage/storm_prediction/data/storms.csv"
OUTPUT_DIR    = "/home/dev/databattle2026-meteorage/storm_prediction/data/storms_output/"
TOP_N         = 20   # top N features à afficher dans les graphiques

# Colonnes à exclure des features
META_COLS = ["airport", "storm_id", "start_time", "end_time"]
TARGET_COLS = ["n_lightnings"]  # servira à construire le label


def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le CSV, crée le label binaire, renvoie (X, y)."""
    df = pd.read_csv(path)

    # ---------- Label binaire ----------
    # Orage "sévère" = n_lightnings > médiane
    median_lightnings = df["n_lightnings"].median()
    df["label_binary"] = (df["n_lightnings"] > median_lightnings).astype(int)
    print(f"  Seuil n_lightnings (médiane) : {median_lightnings:.0f}")

    # ---------- Sélection des features ----------
    # On garde uniquement les colonnes numériques agrégées (_mean, _min, _max, _std)
    # + les caractéristiques lightning (amp_mean, amp_std, dist_mean, icloud_ratio, cg_ratio)
    # + duration_min, et les coords/altitude
    exclude_prefixes = ("Q", "NUM_POSTE", "NOM_USUEL", "AAAAMMJJHH")
    exclude_cols = set(META_COLS + TARGET_COLS + ["label_binary"])

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        # Exclure les colonnes de qualité (Q*_first) et métadonnées station
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        # Exclure les colonnes _first qui sont des métadonnées station
        if c.endswith("_first"):
            continue
        feature_cols.append(c)

    # Ne garder que les colonnes numériques
    X = df[feature_cols].select_dtypes(include=[np.number])

    # 1) Supprimer les colonnes avec > 50 % de NaN
    nan_ratio = X.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
    if cols_to_drop:
        print(f"  {len(cols_to_drop)} colonnes supprimées (>{50}% NaN)")
    X = X.drop(columns=cols_to_drop)

    # 2) Remplir les NaN restants par la médiane de chaque colonne
    X = X.fillna(X.median())

    y = df.loc[X.index, "label_binary"]

    print(f"  {len(X):,} lignes — {X.shape[1]} features — taux label=1 : {y.mean():.1%}")
    return X, y


def plot_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Corrélation Pearson + Mutual Information, puis graphique."""
    corr = X.corrwith(y).rename("pearson_corr")
    mi = pd.Series(
        mutual_info_classif(X, y, random_state=42),
        index=X.columns,
        name="mutual_info"
    )

    ranking = pd.concat([corr.abs().rename("abs_pearson"), mi], axis=1)
    ranking["combined_rank"] = (
        ranking["abs_pearson"].rank(ascending=False) +
        ranking["mutual_info"].rank(ascending=False)
    )
    ranking = ranking.sort_values("combined_rank")
    ranking.to_csv(os.path.join(OUTPUT_DIR, "feature_ranking.csv"))

    top = ranking.head(TOP_N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Pearson
    colors = ["steelblue" if v >= 0 else "tomato"
              for v in corr.loc[top.index]]
    axes[0].barh(top.index[::-1], corr.loc[top.index[::-1]], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"Corrélation Pearson avec label (top {TOP_N})")
    axes[0].set_xlabel("Corrélation")

    # Mutual Information
    axes[1].barh(top.index[::-1], mi.loc[top.index[::-1]], color="mediumseagreen")
    axes[1].set_title(f"Mutual Information avec label (top {TOP_N})")
    axes[1].set_xlabel("MI score")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_feature_importance.png"), dpi=150)
    plt.close()
    print(f"  → 1_feature_importance.png")

    return ranking


def plot_correlation_matrix(X: pd.DataFrame, ranking: pd.DataFrame):
    """Heatmap de corrélation inter-features (top N)."""
    top_features = ranking.head(TOP_N).index.tolist()
    corr_matrix = X[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(f"Matrice de corrélation — top {TOP_N} features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_correlation_matrix.png"), dpi=150)
    plt.close()
    print(f"  → 2_correlation_matrix.png")


def plot_distributions(X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame):
    """Histogrammes des top features, séparés par label."""
    top_features = ranking.head(TOP_N).index.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X[top_features]),
        columns=top_features,
        index=X.index
    )
    y = y.loc[X.index]

    n_cols = 4
    n_rows = int(np.ceil(TOP_N / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        for label, color, name in [(0, "steelblue", "Peu sévère"),
                                   (1, "tomato",    "Sévère")]:
            vals = X_scaled.loc[y == label, feat].dropna()
            ax.hist(vals, bins=40, alpha=0.6, color=color,
                    label=name, density=True)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distribution des top {TOP_N} features par label (normalisées)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_feature_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 3_feature_distributions.png")


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
    print(ranking.head(10)[["abs_pearson", "mutual_info"]].to_string())
    print(f"\nRésultats → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
