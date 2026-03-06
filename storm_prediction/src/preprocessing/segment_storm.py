import pandas as pd

"""
Segmente les éclairs en orages distincts par aéroport.
Règle : un nouvel orage commence si le gap depuis le dernier éclair > GAP_MINUTES.

Input  : data/raw/data.csv
Output : data/processed/storms.csv
"""

# Config
RAW_PATH       = "../data_enrichies/enrichi.csv"
OUTPUT_PATH    = "../output/processed/processed_enrichi.csv"
GAP_MINUTES    = 30
MIN_LIGHTNINGS = 3  # Si orages < N éclairs => on l'ignore


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["airport", "date"]).reset_index(drop=True)
    return df

def assign_storm_ids(df: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    df = df.copy()
    gap = pd.Timedelta(minutes=gap_minutes)

    df["prev_date"] = df.groupby("airport")["date"].shift(1)
    df["time_since_prev"] = df["date"] - df["prev_date"]

    # Nouveau segment si gap > seuil OU premier éclair de cet aéroport
    new_storm = (df["time_since_prev"] > gap) | (df["time_since_prev"].isna())
    df["storm_id"] = new_storm.cumsum().astype(str)

    df["storm_id"] = df["airport"] + "_" + df.groupby("airport")["storm_id"].transform(
        lambda x: pd.factorize(x)[0] + 1
    ).astype(str).str.zfill(4)

    df = df.drop(columns=["prev_date", "time_since_prev"])
    return df


def build_storm_summary(df: pd.DataFrame) -> pd.DataFrame:

    MARINE_SNOW_COLS = {
        "TMER","VVMER","ETATMER","DIRHOULE","HVAGUE","PVAGUE",
        "HNEIGEF","NEIGETOT","TSNEIGE","TUBENEIGE","HNEIGEFI3","HNEIGEFI1","ESNEIGE","CHARGENEIGE",
        "QTMER","QVVMER","QETATMER","QDIRHOULE","QHVAGUE","QPVAGUE",
        "QHNEIGEF","QNEIGETOT","QTSNEIGE","QTUBENEIGE","QHNEIGEFI3","QHNEIGEFI1","QESNEIGE","QCHARGENEIGE",
    }

    STATIC_COLS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH"]

    LIGHTNING_COLS = {
        "airport", "storm_id", "date", "lightning_id", "lightning_airport_id",
        "airport_alert_id", "amplitude", "maxis", "icloud", "dist", "azimuth",
        "is_last_lightning_cloud_ground", "lon", "lat",
    }

    all_cols = set(df.columns)
    exclude  = LIGHTNING_COLS | MARINE_SNOW_COLS | set(STATIC_COLS)

    quality_cols    = [c for c in all_cols if c.startswith("Q") and c not in exclude]
    continuous_cols = [
        c for c in all_cols
        if c not in exclude and c not in quality_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    agg_dict = {
        "date":      ["min", "max", "count"],
        "amplitude": ["mean", "std"],
        "dist":      ["mean"],
        "icloud":    ["mean"],
        "is_last_lightning_cloud_ground": ["mean"],
    }

    for c in STATIC_COLS:
        if c in all_cols:
            agg_dict[c] = "first"

    for c in quality_cols:
        if c in all_cols:
            agg_dict[c] = "first"

    for c in continuous_cols:
        if c in all_cols:
            agg_dict[c] = ["mean", "min", "max", "std"]

    summary = df.groupby(["airport", "storm_id"]).agg(agg_dict)

    # Flatten MultiIndex — from: stackoverflow.com/a/50558529
    summary.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c
                       for c in summary.columns]
    summary = summary.reset_index()

    summary = summary.rename(columns={
        "date_min":   "start_time",
        "date_max":   "end_time",
        "date_count": "n_lightnings",
        "amplitude_mean": "amp_mean",
        "amplitude_std":  "amp_std",
        "dist_mean":      "dist_mean",
        "icloud_mean":    "icloud_ratio",
        "is_last_lightning_cloud_ground_mean": "cg_ratio",
    })

    summary["duration_min"] = (
        (summary["end_time"] - summary["start_time"]).dt.total_seconds() / 60
    ).round(1)
    return summary


def filter_storms(summary: pd.DataFrame, min_lightnings: int) -> pd.DataFrame:
    before = len(summary)
    summary = summary[summary["n_lightnings"] >= min_lightnings].reset_index(drop=True)
    print(f"Orages filtrés (< {min_lightnings} éclairs) : {before - len(summary)} supprimés")
    return summary


def main():
    df = load_data(RAW_PATH)
    print(f"  {len(df):,} éclairs chargés — {df['airport'].nunique()} aéroports")
    df = assign_storm_ids(df, GAP_MINUTES)
    summary = build_storm_summary(df)
    summary = filter_storms(summary, MIN_LIGHTNINGS)
    print(f"  {len(summary):,} orages identifiés")
    summary.to_csv(OUTPUT_PATH, index=False)
    enriched_path = OUTPUT_PATH
    df.to_csv(enriched_path, index=False)
    print(f"Éclairs enrichis → {enriched_path}")

if __name__ == "__main__":
    main()
