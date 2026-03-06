"""
Survival Analysis sur la prédiction de fin d'orage.
Utilise le dernier snapshot de chaque orage comme point de décision unique.

Modèles :
  - Weibull AFT (parametric) — donne directement un percentile 95% → borne supérieure étape 2
  - Cox PH (semi-parametric)  — baseline robuste

Input  : data/processed/features.csv
Output : output/survival/metrics.csv + plots

from : https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import WeibullAFTFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "../../data/processed/features.csv"
OUTPUT_DIR    = "output/survival"

EXCLUDE_COLS  = [
    "storm_id", "airport", "snapshot_time",
    "label_binary", "time_to_end_min", "event",
    "time_since_last", "storm_duration_min",
]
DROP_LOW_SIGNAL = [
    "w5_dist_mean", "w10_dist_mean", "w20_dist_mean",
    "w5_loc_error_mean", "w10_loc_error_mean", "w20_loc_error_mean",
    "w5_loc_error_trend", "w10_loc_error_trend", "w20_loc_error_trend",
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
PERCENTILE  = 95    # borne supérieure : "l'orage se termine avant T à 95%"
# ─────────────────────────────────────────────────────────────────────────────


# ── Chargement ────────────────────────────────────────────────────────────────

def load_last_snapshots(path: str):
    """
    Prend un snapshot aléatoire dans la première moitié de chaque orage.
    from : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
    """
    df = pd.read_csv(path, parse_dates=["snapshot_time"])
    df = df[df["snapshot_time"].dt.year >= 2017]
    df = df.sort_values(["storm_id", "snapshot_time"])

    def sample_first_half(group):
        half = max(1, len(group) // 2)
        return group.iloc[:half].sample(1, random_state=42)

    df = df.groupby("storm_id", group_keys=False).apply(sample_first_half)
    df = df.dropna().reset_index(drop=True)

    print(f"  {len(df):,} orages — durée médiane restante : {df['time_to_end_min'].median():.0f} min")
    print(f"  time_to_end_min : min={df['time_to_end_min'].min():.1f} max={df['time_to_end_min'].max():.1f}")
    return df

def split_temporal(df: pd.DataFrame):
    df = df.sort_values("snapshot_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": df.iloc[:train_end],
        "val":   df.iloc[train_end:val_end],
        "test":  df.iloc[val_end:],
    }
    for name, s in splits.items():
        print(f"  {name:5s} : {len(s):,} orages")
    return splits


def prepare_xy(df: pd.DataFrame, feature_cols: list, scaler=None):
    """
    Retourne un DataFrame lifelines-compatible :
    features + duration_col + event_col
    from : https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
    """
    X = df[feature_cols].copy()

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    X_scaled["time_to_end_min"] = df["time_to_end_min"].values
    X_scaled["event"]           = df["event"].values

    # lifelines requiert duration > 0
    X_scaled["time_to_end_min"] = X_scaled["time_to_end_min"].clip(lower=0.1)

    return X_scaled, scaler


# ── Métriques ─────────────────────────────────────────────────────────────────

def evaluate_model(fitter, df_test: pd.DataFrame, feature_cols: list,
                   model_name: str) -> dict:
    """
    C-index + erreur sur la borne percentile.
    from : https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
    """
    # C-index
    # from : https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index
    if hasattr(fitter, "predict_median"):
        pred_median = fitter.predict_median(df_test[feature_cols])
    else:
        pred_median = fitter.predict_partial_hazard(df_test[feature_cols])

    c_index = concordance_index(
        df_test["time_to_end_min"],
        -pred_median if hasattr(fitter, "predict_median") else pred_median,
        df_test["event"]
    )

    # Borne supérieure : percentile 95% de la survie
    if hasattr(fitter, "predict_percentile"):
        upper_bound = fitter.predict_percentile(df_test[feature_cols],
                                                p=PERCENTILE / 100)
        mae_bound = (upper_bound - df_test["time_to_end_min"]).abs().median()
        coverage  = (upper_bound >= df_test["time_to_end_min"]).mean()
    else:
        mae_bound = np.nan
        coverage  = np.nan

    return {
        "model":         model_name,
        "c_index":       round(c_index, 4),
        "mae_bound_min": round(mae_bound, 2) if not np.isnan(mae_bound) else np.nan,
        "coverage_95p":  round(coverage, 4) if not np.isnan(coverage) else np.nan,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_survival_curves(fitter, df_test: pd.DataFrame, feature_cols: list,
                         model_name: str):
    """Courbes de survie pour quelques orages du test set."""
    sample = df_test.sample(min(5, len(df_test)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(fitter, "predict_survival_function"):
        sf = fitter.predict_survival_function(sample[feature_cols])
        for col in sf.columns:
            ax.plot(sf.index, sf[col], alpha=0.7)

    ax.axhline(0.05, color="red", linestyle="--", label="seuil 95%")
    ax.set_xlabel("Temps restant (min)")
    ax.set_ylabel("P(orage encore actif)")
    ax.set_title(f"Courbes de survie — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"survival_curves_{model_name}.png"), dpi=150)
    plt.close()
    print(f"  → survival_curves_{model_name}.png")


def plot_bound_vs_actual(fitter, df_test: pd.DataFrame, feature_cols: list,
                         model_name: str):
    """Borne prédite vs temps réel restant."""
    if not hasattr(fitter, "predict_percentile"):
        return

    upper = fitter.predict_percentile(df_test[feature_cols], p=PERCENTILE / 100)
    actual = df_test["time_to_end_min"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, upper, alpha=0.3, s=10, color="steelblue")
    ax.plot([0, actual.max()], [0, actual.max()], "r--", label="parfait")
    ax.set_xlabel("Temps réel restant (min)")
    ax.set_ylabel(f"Borne prédite P{PERCENTILE} (min)")
    ax.set_title(f"Borne supérieure prédite vs réelle — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"bound_vs_actual_{model_name}.png"), dpi=150)
    plt.close()
    print(f"  → bound_vs_actual_{model_name}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement (dernier snapshot par orage)...")
    df = load_last_snapshots(FEATURES_PATH)

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and c not in DROP_LOW_SIGNAL]

    print("\nSplit temporel...")
    splits = split_temporal(df)

    df_train = splits["train"]
    df_test  = pd.concat([splits["val"], splits["test"]])  # val+test pour évaluation

    print(f"\nPréparation features ({len(feature_cols)} features)...")
    df_train_fit, scaler = prepare_xy(df_train, feature_cols)
    df_test_fit,  _      = prepare_xy(df_test,  feature_cols, scaler=scaler)

    all_metrics = []

    # ── Weibull AFT ───────────────────────────────────────────────────────────
    print("\nWeibull AFT...")
    weibull = WeibullAFTFitter(penalizer=0.1)
    weibull.fit(df_train_fit, duration_col="time_to_end_min", event_col="event")

    m = evaluate_model(weibull, df_test_fit, feature_cols, "WeibullAFT")
    all_metrics.append(m)
    print(f"  C-index     : {m['c_index']}")
    print(f"  MAE borne   : {m['mae_bound_min']} min")
    print(f"  Coverage 95%: {m['coverage_95p']:.1%}")

    plot_survival_curves(weibull, df_test_fit, feature_cols, "WeibullAFT")
    plot_bound_vs_actual(weibull, df_test_fit, feature_cols, "WeibullAFT")

    # ── Cox PH ───────────────────────────────────────────────────────────────
    print("\nCox PH...")
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(df_train_fit, duration_col="time_to_end_min", event_col="event")

    m = evaluate_model(cox, df_test_fit, feature_cols, "CoxPH")
    all_metrics.append(m)
    print(f"  C-index     : {m['c_index']}")

    plot_survival_curves(cox, df_test_fit, feature_cols, "CoxPH")

    # ── Résumé ────────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    print(f"\n{'='*40}")
    print(metrics_df.to_string(index=False))
    print(f"\nRésultats → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
