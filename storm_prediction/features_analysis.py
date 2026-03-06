import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

"""
Analyse exploratoire des features avant benchmark.
Produit des graphiques dans output/feature_analysis/
Input  : data/processed/features.csv
Output : output/feature_analysis/*.png + feature_ranking.csv
"""

FEATURES_PATH = "data/processed/features.csv"
OUTPUT_DIR    = "output/feature_analysis"
TOP_N         = 20   # top N features à afficher dans les graphiques
NON_FEATURE_COLS = ["storm_id", "airport", "snapshot_time",
                    "label_binary", "time_to_end_min", "event"]

def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["label_binary"]
    print(f"  {len(df):,} lignes — {len(feature_cols)} features — taux label=1 : {y.mean():.1%}")
    return X, y

def plot_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    corr = X.corrwith(y).rename("pearson_corr")
    mi   = pd.Series(
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
    top_features = ranking.head(TOP_N).index.tolist()
    corr_matrix  = X[top_features].corr()

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
    top_features = ranking.head(TOP_N).index.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[top_features]),
                            columns=top_features,
                            index=X.index)
    y = y.loc[X.index]

    n_cols = 4
    n_rows = int(np.ceil(TOP_N / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        for label, color, name in [(0, "steelblue", "Orage en cours"),
                                   (1, "tomato",    "Fin imminente")]:
            vals = X_scaled.loc[y == label, feat].dropna()
            ax.hist(vals, bins=40, alpha=0.6, color=color,
                    label=name, density=True)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribution des top {TOP_N} features par label (valeurs normalisées)",
                 fontsize=12, y=1.01)
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
    print(f"\nRésultats → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
