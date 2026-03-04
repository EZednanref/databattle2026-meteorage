# train.py
# End-to-end training pipeline.
# Trains LightGBM + LSTM, calibrates the ensemble, and saves all artifacts.
#
# Usage:
#   python train.py --data_path data/lightning.parquet --airport Bron
#   python train.py --data_path data/lightning.parquet  # trains all airports

import argparse
import logging
import os
import sys

import numpy as np

from config import CFG, AIRPORTS
from data_loader import load_data, split_temporal
from features import build_features, build_sequences, save_scaler
from model_lgbm import LightGBMModel
from model_lstm import LSTMModel
from model_ensemble import EnsembleModel

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs(CFG.paths.log_dir, exist_ok=True)
os.makedirs(CFG.paths.model_dir, exist_ok=True)
os.makedirs(CFG.paths.output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(CFG.paths.log_dir, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_airport(data_path: str, airport: str) -> None:
    """Full training pipeline for one airport."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Training pipeline for airport: {airport}")
    logger.info(f"{'=' * 60}")

    # ---- 1. Load and split data ----
    df = load_data(data_path, airport=airport)
    train_df, val_df, test_df = split_temporal(df)

    # ---- 2. Feature engineering ----
    logger.info("Building training features...")
    train_feat, X_train, y_train, scaler = build_features(train_df, fit_scaler=True)

    logger.info("Building validation features...")
    val_feat, X_val, y_val, _ = build_features(val_df, scaler=scaler)

    logger.info("Building test features...")
    test_feat, X_test, y_test, _ = build_features(test_df, scaler=scaler)

    # Save scaler
    scaler_path = os.path.join(CFG.paths.model_dir, f"scaler_{airport}.pkl")
    save_scaler(scaler, scaler_path)

    # ---- 3. LSTM sequences ----
    logger.info("Building LSTM sequences...")
    X_train_seq, y_train_seq, train_seq_idx = build_sequences(train_feat, X_train)
    X_val_seq, y_val_seq, val_seq_idx = build_sequences(val_feat, X_val)
    X_test_seq, y_test_seq, test_seq_idx = build_sequences(test_feat, X_test)

    # Align flat features with sequence indices (LSTM uses same rows)
    X_train_flat = X_train[train_seq_idx]
    y_train_flat = y_train[train_seq_idx]
    X_val_flat = X_val[val_seq_idx]
    y_val_flat = y_val[val_seq_idx]

    # ---- 4. Train LightGBM ----
    lgbm = LightGBMModel()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm.save(CFG.paths.model_path(airport, "lgbm"))

    # Log top features
    top_features = dict(list(lgbm.feature_importance().items())[:10])
    logger.info(f"Top 10 LightGBM features:\n{_fmt_dict(top_features)}")

    # ---- 5. Train LSTM ----
    lstm = LSTMModel()
    lstm.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    lstm.save(CFG.paths.lstm_path(airport))

    # ---- 6. Build and calibrate ensemble ----
    ensemble = EnsembleModel(lgbm, lstm)
    ensemble.fit_calibration(X_val_flat, X_val_seq, y_val_flat)
    ensemble.save(CFG.paths.ensemble_path(airport))

    # ---- 7. Quick test-set evaluation ----
    logger.info("Quick test set evaluation...")
    X_test_flat = X_test[test_seq_idx]
    y_test_flat = y_test[test_seq_idx]

    proba, lower, upper = ensemble.predict(X_test_flat, X_test_seq)

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    roc = roc_auc_score(y_test_flat, proba)
    ap = average_precision_score(y_test_flat, proba)
    brier = brier_score_loss(y_test_flat, proba)
    ci_width = (upper - lower).mean()

    logger.info(
        f"[{airport}] Test results — "
        f"ROC-AUC: {roc:.3f} | "
        f"AvgPrecision: {ap:.3f} | "
        f"Brier: {brier:.4f} | "
        f"Mean CI width: {ci_width:.3f}"
    )
    logger.info(f"Training complete for {airport}. Models saved to '{CFG.paths.model_dir}/'")


def _fmt_dict(d: dict) -> str:
    return "\n".join(f"  {k}: {v:.1f}" for k, v in d.items())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train thunderstorm end prediction model")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to lightning data file (.parquet or .csv)"
    )
    parser.add_argument(
        "--airport", type=str, default=None,
        choices=list(AIRPORTS.keys()),
        help="Airport to train on. If not specified, trains all airports."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    airports = [args.airport] if args.airport else list(AIRPORTS.keys())

    for airport in airports:
        try:
            train_airport(args.data_path, airport)
        except Exception as e:
            logger.error(f"Training failed for {airport}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
