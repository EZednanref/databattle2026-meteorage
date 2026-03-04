# features.py
# Builds the feature matrix from raw lightning data.
# All features are strictly backward-looking (no future leakage).

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                   fit_scaler: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    """
    Build feature matrix from raw lightning DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw lightning data (one row per strike), sorted by (airport_alert_id, date).
    scaler : StandardScaler, optional
        Pre-fitted scaler (use during inference / test). If None and fit_scaler=True,
        a new scaler is fitted on this data.
    fit_scaler : bool
        If True, fit a new scaler on the output features.

    Returns
    -------
    feature_df : pd.DataFrame
        Feature matrix (one row per CG strike inside 20 km).
    X : np.ndarray  shape (n_samples, n_features)
        Scaled feature array ready for models.
    y : np.ndarray  shape (n_samples,)
        Binary target: 1 = last CG strike of alert (storm ended), 0 = not last.
    scaler : StandardScaler
    """
    logger.info("Building features...")

    # Work on CG strikes within 20 km only (where labels are defined)
    cg_inner = df[
        (df["icloud"] == False) &
        (df["dist"] <= 20.0) &
        (df["is_last_lightning_cloud_ground"].notna())
    ].copy()

    # Keep ALL strikes (including IC and outer radius) for computing context features
    all_strikes = df.copy()

    rows = []
    for alert_id, alert_group in cg_inner.groupby("airport_alert_id"):
        alert_all = all_strikes[all_strikes["airport_alert_id"] == alert_id]
        alert_rows = _compute_alert_features(alert_group, alert_all)
        rows.append(alert_rows)

    if not rows:
        raise ValueError("No valid alerts found after filtering.")

    feature_df = pd.concat(rows, ignore_index=True)

    # Target
    y = feature_df["is_last_lightning_cloud_ground"].astype(int).values

    # Drop non-feature columns
    meta_cols = ["airport_alert_id", "lightning_airport_id", "date",
                 "is_last_lightning_cloud_ground", "minutes_to_end"]
    feat_cols = CFG.features.feature_names
    # Keep only columns that exist (safety)
    feat_cols = [c for c in feat_cols if c in feature_df.columns]
    X_raw = feature_df[feat_cols].values.astype(np.float32)

    # Replace any NaN/inf introduced during feature computation
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    elif scaler is not None:
        X = scaler.transform(X_raw)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

    logger.info(
        f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features | "
        f"Positive rate: {y.mean():.3%}"
    )
    return feature_df, X, y, scaler


def build_sequences(feature_df: pd.DataFrame, X: np.ndarray,
                    seq_len: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM input sequences: for each CG strike at position i,
    return the preceding `seq_len` feature vectors as a 3D tensor.

    Returns
    -------
    X_seq : np.ndarray  shape (n_samples, seq_len, n_features)
    y_seq : np.ndarray  shape (n_samples,)
    idx   : np.ndarray  indices in feature_df (for alignment with LightGBM preds)
    """
    seq_len = seq_len or CFG.features.lstm_seq_len
    sequences, labels, indices = [], [], []

    n_features = X.shape[1]
    for alert_id, group in feature_df.groupby("airport_alert_id"):
        idx = group.index.tolist()
        for i in range(len(idx)):
            start = max(0, i + 1 - seq_len)
            seq = X[idx[start:i + 1]]
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), n_features), dtype=np.float32)
                seq = np.vstack([pad, seq])
            seq = seq[:seq_len]
            sequences.append(seq)
            labels.append(feature_df.loc[idx[i], "is_last_lightning_cloud_ground"])
            indices.append(idx[i])

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.int64)
    idx_arr = np.array(indices)
    return X_seq, y_seq, idx_arr


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Per-alert feature computation
# ---------------------------------------------------------------------------

def _compute_alert_features(alert_cg: pd.DataFrame,
                             alert_all: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every CG strike in one alert."""
    rows = []
    alert_start = alert_all["date"].min()
    alert_cg = alert_cg.sort_values("date").reset_index(drop=True)
    alert_all_cg = alert_all[alert_all["icloud"] == False].sort_values("date")

    for i, strike in alert_cg.iterrows():
        t = strike["date"]
        features = {"date": t,
                    "airport_alert_id": strike["airport_alert_id"],
                    "lightning_airport_id": strike["lightning_airport_id"],
                    "is_last_lightning_cloud_ground": strike["is_last_lightning_cloud_ground"]}

        # Minutes to end (for evaluation; not used as model feature)
        features["minutes_to_end"] = np.nan  # filled post-hoc if needed

        # ---- Windows ----
        for w in CFG.features.windows_minutes:
            win_all = alert_all[
                (alert_all["date"] <= t) &
                (alert_all["date"] > t - pd.Timedelta(minutes=w))
            ]
            win_cg = win_all[win_all["icloud"] == False]

            features[f"n_cg_{w}min"] = len(win_cg)
            features[f"cg_rate_{w}min"] = len(win_cg) / w
            if w == 10:
                features["n_ic_10min"] = len(win_all) - len(win_cg)
                features["n_total_10min"] = len(win_all)

        # ---- Rate of change ----
        features["rate_delta_5_10"] = (
            features.get("cg_rate_5min", 0) - features.get("cg_rate_10min", 0)
        )
        features["rate_delta_10_20"] = (
            features.get("cg_rate_10min", 0) - features.get("cg_rate_20min", 0)
        )

        # ---- Inter-strike timing ----
        past_cg = alert_all_cg[alert_all_cg["date"] < t]
        if len(past_cg) > 0:
            features["time_since_last_cg"] = (
                t - past_cg["date"].iloc[-1]
            ).total_seconds() / 60
        else:
            features["time_since_last_cg"] = 0.0

        win10_cg = alert_all_cg[
            (alert_all_cg["date"] <= t) &
            (alert_all_cg["date"] > t - pd.Timedelta(minutes=10))
        ]
        isi_values = win10_cg["date"].sort_values().diff().dt.total_seconds().dropna() / 60
        features["mean_isi_cg_10min"] = isi_values.mean() if len(isi_values) > 0 else 0.0

        # ISI trend: compare last 3 vs first 3 intervals in window
        if len(isi_values) >= 6:
            features["isi_trend"] = isi_values.iloc[-3:].mean() - isi_values.iloc[:3].mean()
        else:
            features["isi_trend"] = 0.0

        # ---- Spatial: distance ----
        if len(win10_cg) > 0:
            features["mean_dist_cg_10min"] = win10_cg["dist"].mean()
            features["min_dist_cg_10min"] = win10_cg["dist"].min()
        else:
            features["mean_dist_cg_10min"] = 20.0
            features["min_dist_cg_10min"] = 20.0

        win20_cg = alert_all_cg[
            (alert_all_cg["date"] <= t) &
            (alert_all_cg["date"] > t - pd.Timedelta(minutes=20))
        ]
        features["mean_dist_cg_20min"] = win20_cg["dist"].mean() if len(win20_cg) > 0 else 20.0

        # Distance trend (positive = moving away from airport)
        if len(win20_cg) >= 6:
            features["dist_trend_cg"] = (
                win20_cg["dist"].iloc[-3:].mean() - win20_cg["dist"].iloc[:3].mean()
            )
        else:
            features["dist_trend_cg"] = 0.0

        # ---- Spatial: spread ----
        features["dist_std_cg_10min"] = win10_cg["dist"].std() if len(win10_cg) > 1 else 0.0
        features["azimuth_std_cg_10min"] = _circular_std(win10_cg["azimuth"]) if len(win10_cg) > 1 else 0.0

        # ---- Amplitude ----
        amp10 = win10_cg["amplitude"].abs()
        features["mean_abs_amp_cg_10min"] = amp10.mean() if len(amp10) > 0 else 0.0
        features["max_abs_amp_cg_10min"] = amp10.max() if len(amp10) > 0 else 0.0

        if len(win20_cg) >= 6:
            amp20 = win20_cg["amplitude"].abs()
            features["amp_trend_cg"] = (
                amp20.iloc[-3:].mean() - amp20.iloc[:3].mean()
            )
        else:
            features["amp_trend_cg"] = 0.0

        # ---- Alert-level context ----
        features["alert_duration_min"] = (t - alert_start).total_seconds() / 60
        features["total_cg_in_alert"] = len(
            alert_all_cg[alert_all_cg["date"] <= t]
        )

        # ---- Temporal context (cyclic encoding) ----
        features["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
        features["month_sin"] = np.sin(2 * np.pi * t.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * t.month / 12)

        rows.append(features)

    return pd.DataFrame(rows)


def _circular_std(angles: pd.Series) -> float:
    """Compute circular standard deviation of angles in degrees."""
    if len(angles) < 2:
        return 0.0
    rad = np.deg2rad(angles.dropna())
    R = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
    return float(np.sqrt(-2 * np.log(R + 1e-9)))
