# config.py
# Central configuration for all hyperparameters, paths, and settings.

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os

# ---------------------------------------------------------------------------
# Airport metadata
# ---------------------------------------------------------------------------

AIRPORTS: Dict[str, Dict] = {
    "Bastia":   {"lon": 9.4837,  "lat": 42.5527},
    "Ajaccio":  {"lon": 8.8029,  "lat": 41.9236},
    "Nantes":   {"lon": -1.6107, "lat": 47.1532},
    "Pise":     {"lon": 10.399,  "lat": 43.695},
    "Biarritz": {"lon": -1.524,  "lat": 43.4683},
}

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Radius used to filter lightning strikes (km)
    alert_radius_km: float = 20.0
    full_radius_km: float = 30.0

    # Exclude Pisa 2016 intra-cloud data (different recording system)
    exclude_pisa_2016_icloud: bool = True

    # Temporal train/val/test split years
    train_end_year: int = 2019
    val_year: int = 2020
    test_start_year: int = 2021

    # Minimum number of CG strikes for an alert to be included in training
    min_cg_strikes_per_alert: int = 3

# ---------------------------------------------------------------------------
# Feature engineering settings
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    # Rolling windows (minutes) for computing strike rates and stats
    windows_minutes: List[int] = field(default_factory=lambda: [5, 10, 20, 30])

    # Sequence length (number of past strikes) for LSTM
    lstm_seq_len: int = 20

    # Features used by both LightGBM and LSTM
    feature_names: List[str] = field(default_factory=lambda: [
        # --- Strike counts ---
        "n_cg_5min",
        "n_cg_10min",
        "n_cg_20min",
        "n_cg_30min",
        "n_ic_10min",
        "n_total_10min",

        # --- Rates ---
        "cg_rate_5min",
        "cg_rate_10min",
        "cg_rate_20min",

        # --- Rate of change (deceleration signal) ---
        "rate_delta_5_10",      # cg_rate_5min - cg_rate_10min
        "rate_delta_10_20",     # cg_rate_10min - cg_rate_20min

        # --- Inter-strike timing ---
        "time_since_last_cg",   # minutes since last CG strike
        "mean_isi_cg_10min",    # mean inter-strike interval (CG, last 10 min)
        "isi_trend",            # is interval growing? (positive = slowing down)

        # --- Spatial: distance ---
        "mean_dist_cg_10min",
        "min_dist_cg_10min",
        "mean_dist_cg_20min",
        "dist_trend_cg",        # positive = storm moving away

        # --- Spatial: spread ---
        "dist_std_cg_10min",
        "azimuth_std_cg_10min",

        # --- Amplitude (energy) ---
        "mean_abs_amp_cg_10min",
        "max_abs_amp_cg_10min",
        "amp_trend_cg",         # positive = weakening

        # --- Alert-level context ---
        "alert_duration_min",   # how long this alert has been active
        "total_cg_in_alert",    # total CG strikes so far in this alert

        # --- Temporal context ---
        "hour_sin",             # time of day encoded cyclically
        "hour_cos",
        "month_sin",            # seasonality
        "month_cos",
    ])

# ---------------------------------------------------------------------------
# LightGBM settings
# ---------------------------------------------------------------------------

@dataclass
class LGBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    verbose: int = -1

# ---------------------------------------------------------------------------
# LSTM settings
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    input_size: int = 28          # = len(feature_names), updated at runtime
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False   # set True for richer representation (slower)

    # Training
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10            # early stopping
    pos_weight: float = 10.0      # BCEWithLogitsLoss weight for positive class

    random_state: int = 42

# ---------------------------------------------------------------------------
# Ensemble settings
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    # Weights for [LightGBM, LSTM] in the weighted average
    weights: Tuple[float, float] = (0.55, 0.45)

    # Calibration method: 'isotonic' or 'sigmoid'
    calibration_method: str = "isotonic"

    # Conformal prediction coverage level (e.g. 0.90 = 90% CI)
    conformal_alpha: float = 0.10

# ---------------------------------------------------------------------------
# Operational thresholds
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    # Probability thresholds for advisory levels
    watch_threshold: float = 0.70       # "Storm possibly ending"
    allclear_threshold: float = 0.92    # "All-clear recommended"

    # Confirmation: alert stays lifted only if probability stays above
    # allclear_threshold for this many consecutive minutes
    confirmation_minutes: int = 10

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    def model_path(self, airport: str, model_type: str) -> str:
        return os.path.join(self.model_dir, f"{model_type}_{airport}.pkl")

    def lstm_path(self, airport: str) -> str:
        return os.path.join(self.model_dir, f"lstm_{airport}.pt")

    def ensemble_path(self, airport: str) -> str:
        return os.path.join(self.model_dir, f"ensemble_{airport}.pkl")

# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Singleton
CFG = Config()
