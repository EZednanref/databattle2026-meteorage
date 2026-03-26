# data_loader.py
# Loads raw lightning data, validates schema, applies exclusions,
# and returns clean DataFrames ready for feature engineering.

import os
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import CFG, AIRPORTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "date", "lon", "lat", "amplitude", "maxis",
    "icloud", "dist", "azimuth",
    "lightning_id", "lightning_airport_id",
    "airport_alert_id", "is_last_lightning_cloud_ground",
]

DTYPES = {
    "amplitude": float,
    "maxis": float,
    "dist": float,
    "azimuth": float,
    "icloud": bool,
    "is_last_lightning_cloud_ground": object,   # can be bool or NaN
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    path: str,
    airport: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and validate a lightning strike dataset.

    Parameters
    ----------
    path : str
        Path to a .parquet or .csv file.
    airport : str, optional
        If provided, filter to this airport only. Must be a key in AIRPORTS.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by (airport_alert_id, date).
    """
    logger.info(f"Loading data from {path}")
    df = _read_file(path)
    df = _validate_schema(df)
    df = _parse_types(df)
    df = _apply_exclusions(df)

    if airport is not None:
        if airport not in AIRPORTS:
            raise ValueError(f"Unknown airport '{airport}'. Choose from {list(AIRPORTS.keys())}")
        if "airport" in df.columns:
            df = df[df["airport"] == airport].copy()
            logger.info(f"Filtered to airport '{airport}': {len(df):,} rows")
        else:
            logger.warning("Column 'airport' not found — skipping airport filter.")

    df = df.sort_values(["airport_alert_id", "date"]).reset_index(drop=True)
    _log_summary(df)
    return df


def split_temporal(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test sets by year (no alert spans two splits).

    Returns
    -------
    train_df, val_df, test_df
    """
    train_end = CFG.data.train_end_year
    val_year  = CFG.data.val_year
    test_start = CFG.data.test_start_year

    year = df["date"].dt.year
    train_df = df[year <= train_end].copy()
    val_df   = df[year == val_year].copy()
    test_df  = df[year >= test_start].copy()

    logger.info(
        f"Temporal split — "
        f"train: {len(train_df):,} rows ({year[year <= train_end].nunique()} years) | "
        f"val: {len(val_df):,} rows | "
        f"test: {len(test_df):,} rows"
    )
    return train_df, val_df, test_df


def get_alert_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per alert with summary statistics.
    Useful for sanity checks and stratified sampling.
    """
    cg = df[df["icloud"] == False]
    summary = (
        cg.groupby("airport_alert_id")
        .agg(
            n_cg_strikes=("lightning_airport_id", "count"),
            alert_start=("date", "min"),
            alert_end=("date", "max"),
            airport=("airport", "first") if "airport" in df.columns else ("dist", "count"),
        )
        .assign(
            duration_min=lambda x: (x["alert_end"] - x["alert_start"]).dt.total_seconds() / 60
        )
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(path, low_memory=False)
    elif ext in (".feather", ".fea"):
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .parquet, .csv, or .feather")


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS + ["airport"]]
    if extra:
        logger.debug(f"Extra columns (kept): {extra}")
    return df


def _parse_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Numeric coercions
    for col, dtype in DTYPES.items():
        if col in df.columns and dtype in (float, int):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean: icloud
    if df["icloud"].dtype != bool:
        df["icloud"] = df["icloud"].map({"True": True, "False": False, True: True, False: False})

    # Boolean: is_last_lightning_cloud_ground (can be NaN for strikes outside 20 km)
    if "is_last_lightning_cloud_ground" in df.columns:
        df["is_last_lightning_cloud_ground"] = df["is_last_lightning_cloud_ground"].map(
            {True: True, False: False, "True": True, "False": False}
        )  # NaN stays NaN

    return df


def _apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # Drop rows with invalid distances or coordinates
    df = df.dropna(subset=["dist", "lon", "lat", "amplitude"])
    df = df[df["dist"] >= 0]
    df = df[df["dist"] <= CFG.data.full_radius_km]

    # Exclude Pisa 2016 intra-cloud data
    if CFG.data.exclude_pisa_2016_icloud and "airport" in df.columns:
        mask = (
            (df["airport"] == "Pisa") &
            (df["date"].dt.year == 2016) &
            (df["icloud"] == True)
        )
        n_excluded = mask.sum()
        df = df[~mask].copy()
        logger.info(f"Excluded {n_excluded:,} Pisa 2016 intra-cloud strikes")

    # Remove alerts with too few CG strikes (not enough signal)
    cg_counts = (
        df[df["icloud"] == False]
        .groupby("airport_alert_id")["lightning_airport_id"]
        .count()
    )
    valid_alerts = cg_counts[cg_counts >= CFG.data.min_cg_strikes_per_alert].index
    df = df[df["airport_alert_id"].isin(valid_alerts)].copy()

    n_after = len(df)
    logger.info(f"Exclusions: {n_before:,} → {n_after:,} rows ({n_before - n_after:,} dropped)")
    return df


def _log_summary(df: pd.DataFrame) -> None:
    n_alerts = df["airport_alert_id"].nunique()
    n_cg = (df["icloud"] == False).sum()
    n_ic = (df["icloud"] == True).sum()
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
    logger.info(
        f"Dataset summary: {len(df):,} strikes | "
        f"{n_alerts:,} alerts | "
        f"{n_cg:,} CG | {n_ic:,} IC | "
        f"{date_range}"
    )
