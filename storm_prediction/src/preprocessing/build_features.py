import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
"""
Construit une feature matrix à partir des éclairs segmentés.
Pour chaque éclair (= point de décision), on calcule des features
sur les N derniers éclairs de l'orage en cours.

Input  : data/raw/data_with_storm_id.csv
         data/processed/storms.csv
Output : data/processed/features.csv
"""
# Config
LIGHTNING_PATH = "../data/raw/data_with_storm_id.csv"
STORMS_PATH    = "../data/processed/storms.csv"
OUTPUT_PATH    = "../data/processed/features.csv"

WINDOW_SIZES   = [5, 10, 20]   # fenêtres glissantes (nombre d'éclairs)
MIN_WINDOW     = 5             # points de décision ignorés si < N éclairs dans l'orage


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    lightnings = pd.read_csv(LIGHTNING_PATH, parse_dates=["date"])
    lightnings = lightnings.sort_values(["storm_id", "date"]).reset_index(drop=True)

    for col in ["dist", "azimuth"]:  # add other float columns as needed
        if col in lightnings.columns:
            lightnings[col] = lightnings[col].astype(np.float32)
    storms = pd.read_csv(STORMS_PATH, parse_dates=["start_time", "end_time"])
    return lightnings, storms


def _slope(series: pd.Series) -> float:
    """Pente linéaire (tendance) d'une série."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=np.float32)
    return float(np.polyfit(x, series.astype(np.float32), 1)[0])


# from : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
def process_storm(storm_id, group, storm_end):
    records = []
    group = group.sort_values("date").reset_index(drop=True)
    storm_start = group["date"].iloc[0]  # ← ajouter
    for i in range(MIN_WINDOW, len(group)):
        window = group.iloc[: i + 1]
        snapshot_time = window["date"].iloc[-1]
        row = {
            "storm_id": storm_id,
            "airport": group["airport"].iloc[0],
            "snapshot_time": snapshot_time,
        }
        for w in WINDOW_SIZES:
            if i + 1 >= w:
                row.update(compute_window_features(window, w))
        row.update(compute_storm_context_features(window, snapshot_time))
        row.update(compute_labels(snapshot_time, storm_end, storm_start))  # ← modifié
        records.append(row)
    return records

def compute_window_features(window: pd.DataFrame, size: int) -> dict:
    """Features calculées sur une fenêtre de `size` éclairs."""
    w = window.tail(size)
    prefix = f"w{size}_"

    inter_gaps = w["date"].diff().dt.total_seconds().dropna() / 60  # en minutes

    return {
        f"{prefix}gap_mean":          inter_gaps.mean() if len(inter_gaps) else np.nan,
        f"{prefix}gap_max":           inter_gaps.max()  if len(inter_gaps) else np.nan,
        f"{prefix}gap_last":          inter_gaps.iloc[-1] if len(inter_gaps) else np.nan,
        f"{prefix}lightning_rate":    size / (inter_gaps.sum() + 1e-6),  # éclairs/min
        f"{prefix}rate_trend":        _slope(
            w["date"].diff().dt.total_seconds().dropna()
        ),  # >0 = ralentissement
        f"{prefix}amp_mean":          w["amplitude"].mean(),
        f"{prefix}amp_std":           w["amplitude"].std(),
        f"{prefix}amp_trend":         _slope(w["amplitude"]),
        f"{prefix}dist_mean":         w["dist"].mean(),
        f"{prefix}dist_trend":        _slope(w["dist"]),  # >0 = cellule qui s'éloigne
        f"{prefix}icloud_ratio":      w["icloud"].mean(),
        f"{prefix}cg_ratio":          w["is_last_lightning_cloud_ground"].mean(),
        f"{prefix}azimuth_std":       w["azimuth"].std(),
        # Qualité de mesure (maxis = erreur de localisation estimée en km)
        # Une erreur croissante peut indiquer une cellule qui s'éloigne / affaiblit
        f"{prefix}loc_error_mean":    w["maxis"].mean() if "maxis" in w.columns else np.nan,
        f"{prefix}loc_error_trend":   _slope(w["maxis"]) if "maxis" in w.columns else np.nan,
    }


def compute_storm_context_features(window: pd.DataFrame, snapshot_time: pd.Timestamp) -> dict:
    storm_age = (snapshot_time - window["date"].min()).total_seconds() / 60
    return {
        "storm_age_min":        storm_age,
        "n_lightnings_so_far":  len(window),
        "time_since_last":      (snapshot_time - window["date"].max()).total_seconds() / 60,
        "global_amp_mean":      window["amplitude"].mean(),
        "global_amp_trend":     _slope(window["amplitude"]),
        "global_dist_trend":    _slope(window["dist"]),
        "global_icloud_ratio":  window["icloud"].mean(),
    }


# from : https://www.sciencedirect.com/science/article/pii/S0169809521003290
# dissipation phase ~last 40% of storm lifetime
DISSIPATION_RATIO = 0.40
def compute_labels(snapshot_time, storm_end, storm_start) -> dict:
    storm_duration = (storm_end - storm_start).total_seconds() / 60
    time_to_end    = (storm_end - snapshot_time).total_seconds() / 60
    return {
        "label_binary":       int(time_to_end <= DISSIPATION_RATIO * storm_duration),
        "time_to_end_min":    max(time_to_end, 0.0),
        "storm_duration_min": storm_duration,
        "event":              1,
    }


def build_features(lightnings, storms):
    storm_end_map = storms.set_index("storm_id")["end_time"].to_dict()
    results = Parallel(n_jobs=-1)(
        delayed(process_storm)(storm_id, group, storm_end_map.get(storm_id))
        for storm_id, group in tqdm(lightnings.groupby("storm_id"))
        if storm_end_map.get(storm_id) is not None
    )
    records = [row for sublist in results for row in sublist]
    return pd.DataFrame(records)


def main():
    print("Chargement...")
    lightnings, storms = load_data()
    print(f"  {len(lightnings):,} éclairs — {lightnings['storm_id'].nunique()} orages")

    features = build_features(lightnings, storms)

    print(f"  {len(features):,} points de décision générés")
    print(f"  Taux label=1 : {features['label_binary'].mean():.1%}")
    print(f"  Colonnes     : {features.shape[1]}")

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Sauvegardé → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
