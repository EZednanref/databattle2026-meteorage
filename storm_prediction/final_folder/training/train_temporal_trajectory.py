import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE

DATA_PATH    = "enrichi.csv"
OUTPUT_DIR   = "model_output/"
ENRICHED_CSV = "../output/data_enrichie_features.csv"
TARGET_COL   = "is_last_lightning_cloud_ground"

# Colonnes à exclure des features ML
META_COLS = [
    "lightning_id", "lightning_airport_id", "airport", "airport_alert_id",
    "storm_id", "date", "icloud",
    "LAT", "LON", "ALTI", "NOM_USUEL", "NUM_POSTE", "AAAAMMJJHH",
    "lon", "lat",       # coords éclair : utilisées pour trajectoire mais pas comme feature
    "azimuth", "amplitude",
]


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def linear_trend_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=3).apply(
        lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 3 else 0,
        raw=True
    )


def time_window_trend(group: pd.DataFrame, minutes: int) -> pd.Series:
    result = pd.Series(0.0, index=group.index)
    times  = group['date']
    dists  = group['dist']
    for idx in group.index:
        t     = times[idx]
        mask  = (times <= t) & (times >= t - pd.Timedelta(minutes=minutes))
        chunk = dists[mask]
        if len(chunk) >= 3:
            result[idx] = np.polyfit(range(len(chunk)), chunk.values, 1)[0]
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['storm_id', 'date']).copy()
    df['date'] = pd.to_datetime(df['date'])
    grp = df.groupby('storm_id')

    # ── Temporelles ──────────────────────────────
    df['lightning_rank']     = grp.cumcount()
    df['time_since_start']   = (df['date'] - grp['date'].transform('first')).dt.total_seconds()
    df['time_since_last']    = grp['date'].transform(
        lambda x: x.diff().dt.total_seconds()
    ).fillna(0)
    df['dist_delta']         = grp['dist'].diff().fillna(0)
    df['dist_rolling_mean']  = grp['dist'].transform(lambda x: x.expanding().mean())
    df['lightning_rate']     = df['lightning_rank'] / (df['time_since_start'] + 1)
    df['maxis_rolling_mean'] = grp['maxis'].transform(lambda x: x.expanding().mean())
    df['n_icloud_so_far']    = grp['icloud'].transform(lambda x: x.expanding().sum())

    # ── Trajectoire (lon/lat = coords de l'éclair) ───
    dt = grp['date'].transform(lambda x: x.diff().dt.total_seconds()).fillna(0)

    df['delta_lon']      = grp['lon'].diff().fillna(0)
    df['delta_lat']      = grp['lat'].diff().fillna(0)
    df['speed']          = np.sqrt(df['delta_lon']**2 + df['delta_lat']**2) / (dt + 1)
    df['bearing']        = np.arctan2(df['delta_lat'], df['delta_lon'])
    df['approach_speed'] = grp['dist'].diff().fillna(0) / (dt + 1)
    df['centroid_lon']   = grp['lon'].transform(lambda x: x.expanding().mean())
    df['centroid_lat']   = grp['lat'].transform(lambda x: x.expanding().mean())
    df['spread']         = np.sqrt(
        (df['lon'] - df['centroid_lon'])**2 +
        (df['lat'] - df['centroid_lat'])**2
    )

    # ── Tendances dist par nb d'éclairs ──────────
    df['dist_trend_n10'] = grp['dist'].transform(lambda x: linear_trend_rolling(x, 10))
    df['dist_trend_n20'] = grp['dist'].transform(lambda x: linear_trend_rolling(x, 20))

    # ── Tendances dist par fenêtre temporelle ────
    print("  Calcul dist_trend_15min...")
    trends_15 = df.groupby('storm_id')[['date', 'dist']].apply(
        lambda g: time_window_trend(g, 15)
    ).reset_index(level=0, drop=True)

    print("  Calcul dist_trend_30min...")
    trends_30 = df.groupby('storm_id')[['date', 'dist']].apply(
        lambda g: time_window_trend(g, 30)
    ).reset_index(level=0, drop=True)

    df['dist_trend_15min'] = trends_15
    df['dist_trend_30min'] = trends_30

    df = df.drop(columns=['delta_lon', 'delta_lat'])
    return df


# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────

def load(path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # df_raw = pd.read_csv(path, low_memory=False)
    #
    # print("  Construction des features temporelles + trajectoire...")
    # df = build_features(df_raw)
    #
    # print(f"  Export CSV enrichi → {ENRICHED_CSV}")
    # print("chargement du CSV enrichi...")
    df = pd.read_csv(ENRICHED_CSV, low_memory=False)
    df.to_csv(ENRICHED_CSV, index=False)

    print("suppr iclouds...")
    df = df[df["icloud"] == False].copy()

    # print("exclusion des colonnes non numériques + features temporelles/trajectoire...")
    exclude = set(META_COLS + [TARGET_COL])
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and not c.startswith("Q")
                    and not c.endswith("_first")]

    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.drop(columns=X.columns[X.isna().mean() > 0.5].tolist())
    X = X.fillna(X.median())

    y      = pd.to_numeric(df.loc[X.index, TARGET_COL], errors="coerce")
    groups = df.loc[X.index, "storm_id"]
    valid  = y.notna() & groups.notna()
    X, y, groups = X.loc[valid], y.loc[valid].astype(int), groups.loc[valid]

    print(f"  {len(X):,} lignes — {X.shape[1]} features — taux label=1 : {y.mean():.1%}")
    print(f"  {groups.nunique():,} orages uniques")
    return X, y, groups


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Chargement + features...")
    X, y, groups = load(DATA_PATH)

    print("Split train/test par storm_id (80/20)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"  Train : {len(X_train):,} lignes ({groups.iloc[train_idx].nunique()} orages)")
    print(f"  Test  : {len(X_test):,} lignes ({groups.iloc[test_idx].nunique()} orages)")
    #
    print("SMOTE 0.5...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Entraînement LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=False,
        metric="average_precision",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train_res, y_train_res)

    print("Évaluation...")
    y_pred  = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)[:, 1]
    roc_auc       = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    importances   = pd.Series(lgbm.feature_importances_, index=X.columns)
    importances   = importances.sort_values(ascending=False).head(15)

    report_lines = [
        "=" * 60,
        "LIGHTGBM + SMOTE 0.5 + GroupSplit + Temporal + Trajectory",
        "=" * 60,
        f"Lignes train : {len(X_train_res):,}  |  test : {len(X_test):,}",
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
        report_lines.append(f"  {feat:<25} {imp}")

    report = "\n".join(report_lines)
    print(report)
    report_path = os.path.join(OUTPUT_DIR, "rapport_lgbm_trajectory.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n→ {report_path}")
    print(f"→ {ENRICHED_CSV}")


if __name__ == "__main__":
    main()
