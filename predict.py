# predict.py
# Real-time inference: given a stream of lightning strikes for an active alert,
# outputs a probability timeline updated after every new strike.
#
# Usage (CLI):
#   python predict.py --model_path models/ensemble_Bron.pkl \
#                     --data_path data/lightning.parquet \
#                     --alert_id 1042
#
# Usage (Python API):
#   from predict import StormPredictor
#   predictor = StormPredictor("models/ensemble_Bron.pkl", airport="Bron")
#   result = predictor.update(new_strike_df)
#   print(result)  # {"time": ..., "probability": 0.82, "lower": 0.74, "upper": 0.89, "advisory": ...}

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import CFG, AIRPORTS
from data_loader import load_data
from features import build_features, build_sequences, load_scaler
from model_ensemble import EnsembleModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-time predictor
# ---------------------------------------------------------------------------

class StormPredictor:
    """
    Stateful predictor for a single ongoing alert.
    Call `update()` after each new lightning strike to get the latest forecast.

    Parameters
    ----------
    model_path : str
        Path to a saved EnsembleModel (.pkl).
    airport : str
        Airport name (must match a key in AIRPORTS).
    """

    def __init__(self, model_path: str, airport: str):
        if airport not in AIRPORTS:
            raise ValueError(f"Unknown airport: {airport}")

        self.airport = airport
        self.ensemble = EnsembleModel.load(model_path)
        scaler_path = model_path.replace("ensemble_", "scaler_").replace(".pkl", ".pkl")
        # Try standard path convention
        import os
        standard_scaler = os.path.join(
            CFG.paths.model_dir, f"scaler_{airport}.pkl"
        )
        self.scaler = load_scaler(standard_scaler)

        # Rolling buffer of all strikes in the current alert
        self._buffer: list = []
        self._alert_id: Optional[str] = None
        self._history: list = []  # list of prediction results

    def reset(self, alert_id: Optional[str] = None) -> None:
        """Start tracking a new alert."""
        self._buffer = []
        self._alert_id = alert_id
        self._history = []
        logger.info(f"Predictor reset for alert_id={alert_id}")

    def update(self, new_strikes: pd.DataFrame) -> Optional[dict]:
        """
        Ingest new lightning strikes and return an updated forecast.

        Parameters
        ----------
        new_strikes : pd.DataFrame
            One or more new rows from the live lightning feed.
            Must contain the same columns as the training data.

        Returns
        -------
        dict with keys:
            time         : UTC timestamp of the latest strike
            probability  : P(storm has ended) in [0, 1]
            lower        : Lower bound of confidence interval
            upper        : Upper bound of confidence interval
            advisory     : Human-readable advisory string
            n_cg_last10  : Number of CG strikes in last 10 min (context)
            last_cg_dist : Distance of most recent CG strike (km)
        """
        # Append to buffer
        self._buffer.append(new_strikes)
        alert_df = pd.concat(self._buffer, ignore_index=True)

        # Only predict on CG strikes inside 20 km
        cg_inner = alert_df[
            (alert_df["icloud"] == False) &
            (alert_df["dist"] <= 20.0)
        ]
        if len(cg_inner) < 1:
            return None

        # Inject a dummy airport_alert_id and label for the feature builder
        alert_df = alert_df.copy()
        if "airport_alert_id" not in alert_df.columns:
            alert_df["airport_alert_id"] = self._alert_id or "live_alert"
        if "is_last_lightning_cloud_ground" not in alert_df.columns:
            alert_df["is_last_lightning_cloud_ground"] = False  # unknown in real-time
        if "icloud" not in alert_df.columns:
            alert_df["icloud"] = False

        try:
            feat_df, X, _, _ = build_features(alert_df, scaler=self.scaler)
            X_seq, _, seq_idx = build_sequences(feat_df, X)
            if len(X_seq) == 0:
                return None

            # Use only the most recent row (last strike)
            x_flat = X[seq_idx[-1]].reshape(1, -1)
            x_seq = X_seq[-1].reshape(1, *X_seq.shape[1:])

            proba, lower, upper = self.ensemble.predict(x_flat, x_seq)
            p = float(proba[0])
            l = float(lower[0])
            u = float(upper[0])

        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None

        t = cg_inner["date"].max()
        cg_10 = cg_inner[cg_inner["date"] > t - pd.Timedelta(minutes=10)]

        result = {
            "time": t,
            "probability": round(p, 4),
            "lower": round(l, 4),
            "upper": round(u, 4),
            "advisory": self.ensemble.advisory(p),
            "n_cg_last10": len(cg_10),
            "last_cg_dist_km": round(float(cg_inner["dist"].iloc[-1]), 1),
        }
        self._history.append(result)
        return result

    def get_timeline(self) -> pd.DataFrame:
        """Return the full prediction history as a formatted table."""
        if not self._history:
            return pd.DataFrame()
        df = pd.DataFrame(self._history)
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%H:%M:%S")
        df["P(ended)"] = (df["probability"] * 100).round(1).astype(str) + "%"
        df["90% CI"] = (
            "[" +
            (df["lower"] * 100).round(1).astype(str) + "% – " +
            (df["upper"] * 100).round(1).astype(str) + "%]"
        )
        return df[["time", "P(ended)", "90% CI", "advisory", "n_cg_last10", "last_cg_dist_km"]]

    def print_timeline(self) -> None:
        """Pretty-print the prediction timeline to stdout."""
        df = self.get_timeline()
        if df.empty:
            print("No predictions yet.")
            return
        print(f"\n{'─' * 85}")
        print(f"{'Time':>10} │ {'P(ended)':>10} │ {'90% CI':>22} │ {'Advisory':<35}")
        print(f"{'─' * 85}")
        for _, row in df.iterrows():
            print(
                f"{row['time']:>10} │ "
                f"{row['P(ended)']:>10} │ "
                f"{row['90% CI']:>22} │ "
                f"{row['advisory']:<35}"
            )
        print(f"{'─' * 85}\n")


# ---------------------------------------------------------------------------
# Batch replay (CLI demo)
# ---------------------------------------------------------------------------

def replay_alert(model_path: str, data_path: str, airport: str, alert_id: int) -> None:
    """
    Replay a historical alert strike by strike and print the probability timeline.
    Useful for demo and model validation.
    """
    df = load_data(data_path, airport=airport)
    alert_df = df[df["airport_alert_id"] == alert_id].sort_values("date")
    if alert_df.empty:
        logger.error(f"Alert {alert_id} not found in data.")
        return

    logger.info(f"Replaying alert {alert_id} ({len(alert_df)} strikes)...")
    predictor = StormPredictor(model_path, airport)
    predictor.reset(alert_id=alert_id)

    # Feed strikes one row at a time (simulating real-time)
    for _, strike in alert_df.iterrows():
        strike_df = strike.to_frame().T.reset_index(drop=True)
        result = predictor.update(strike_df)
        if result:
            print(
                f"[{result['time'].strftime('%H:%M:%S')}] "
                f"P(ended)={result['probability']:.1%} "
                f"CI=[{result['lower']:.1%}–{result['upper']:.1%}] "
                f"| {result['advisory']}"
            )

    predictor.print_timeline()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time thunderstorm end prediction")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--airport", type=str, required=True, choices=list(AIRPORTS.keys()))
    parser.add_argument("--alert_id", type=int, required=True,
                        help="Alert ID to replay from historical data")
    args = parser.parse_args()
    replay_alert(args.model_path, args.data_path, args.airport, args.alert_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
