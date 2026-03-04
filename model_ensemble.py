# model_ensemble.py
# Combines LightGBM + LSTM predictions, applies probability calibration,
# and produces confidence intervals via conformal prediction.

import logging
from typing import Optional, Tuple

import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from config import CFG
from model_lgbm import LightGBMModel
from model_lstm import LSTMModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of LightGBM + LSTM with:
    - Post-hoc probability calibration (isotonic regression)
    - Conformal prediction intervals (split conformal)

    Usage
    -----
    ensemble = EnsembleModel(lgbm_model, lstm_model)
    ensemble.fit_calibration(X_val, X_val_seq, y_val)
    proba, lower, upper = ensemble.predict(X, X_seq)
    """

    def __init__(
        self,
        lgbm_model: LightGBMModel,
        lstm_model: LSTMModel,
        weights: Optional[Tuple[float, float]] = None,
    ):
        self.lgbm = lgbm_model
        self.lstm = lstm_model
        self.weights = weights or CFG.ensemble.weights
        self.calibrator: Optional[IsotonicRegression] = None
        self.conformal_threshold: Optional[float] = None
        self.is_calibrated = False

    # ------------------------------------------------------------------
    def predict_raw(
        self,
        X: np.ndarray,           # (n, n_features) for LightGBM
        X_seq: np.ndarray,       # (n, seq_len, n_features) for LSTM
    ) -> np.ndarray:
        """Weighted average of the two model probabilities (uncalibrated)."""
        p_lgbm = self.lgbm.predict_proba(X)
        p_lstm = self.lstm.predict_proba(X_seq)
        w_lgbm, w_lstm = self.weights
        return w_lgbm * p_lgbm + w_lstm * p_lstm

    # ------------------------------------------------------------------
    def fit_calibration(
        self,
        X_val: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray,
        alpha: Optional[float] = None,
    ) -> "EnsembleModel":
        """
        Fit isotonic regression calibrator + conformal prediction threshold
        on a held-out validation set.

        Parameters
        ----------
        alpha : float
            Miscoverage level for conformal intervals (default from config).
            E.g. alpha=0.10 gives 90% coverage intervals.
        """
        alpha = alpha or CFG.ensemble.conformal_alpha
        raw_proba = self.predict_raw(X_val, X_val_seq)

        # ---- Isotonic calibration ----
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_proba, y_val)

        cal_proba = self.calibrator.predict(raw_proba)
        brier_before = brier_score_loss(y_val, raw_proba)
        brier_after = brier_score_loss(y_val, cal_proba)
        logger.info(
            f"Calibration: Brier score {brier_before:.4f} → {brier_after:.4f}"
        )

        # ---- Split conformal prediction ----
        # Non-conformity score: |y - p_hat|
        nonconf_scores = np.abs(y_val - cal_proba)
        # Threshold = (1-alpha) quantile of nonconformity scores
        n = len(y_val)
        level = np.ceil((n + 1) * (1 - alpha)) / n
        self.conformal_threshold = float(np.quantile(nonconf_scores, level))
        logger.info(
            f"Conformal threshold (alpha={alpha}): {self.conformal_threshold:.4f}"
        )

        self.is_calibrated = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        X_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict calibrated probabilities with confidence intervals.

        Returns
        -------
        proba  : np.ndarray  Calibrated P(storm ended)
        lower  : np.ndarray  Lower bound of confidence interval
        upper  : np.ndarray  Upper bound of confidence interval
        """
        raw = self.predict_raw(X, X_seq)

        if self.is_calibrated and self.calibrator is not None:
            proba = self.calibrator.predict(raw)
        else:
            logger.warning("Model not calibrated — returning raw probabilities.")
            proba = raw

        if self.conformal_threshold is not None:
            lower = np.clip(proba - self.conformal_threshold, 0.0, 1.0)
            upper = np.clip(proba + self.conformal_threshold, 0.0, 1.0)
        else:
            lower = np.zeros_like(proba)
            upper = np.ones_like(proba)

        return proba, lower, upper

    # ------------------------------------------------------------------
    def predict_single(
        self,
        x: np.ndarray,          # (n_features,)
        x_seq: np.ndarray,      # (seq_len, n_features)
    ) -> Tuple[float, float, float]:
        """
        Convenience wrapper for real-time inference on a single sample.

        Returns
        -------
        (probability, lower_bound, upper_bound)
        """
        proba, lower, upper = self.predict(
            x.reshape(1, -1),
            x_seq.reshape(1, *x_seq.shape),
        )
        return float(proba[0]), float(lower[0]), float(upper[0])

    # ------------------------------------------------------------------
    def advisory(self, proba: float) -> str:
        """Map probability to a human-readable operational advisory."""
        t = CFG.thresholds
        if proba >= t.allclear_threshold:
            return "🟢 All-clear recommended"
        elif proba >= t.watch_threshold:
            return "🟡 Storm possibly ending — monitor closely"
        else:
            return "🔴 Storm active — alert maintained"

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Ensemble model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        model = joblib.load(path)
        logger.info(f"Ensemble model loaded from {path}")
        return model
