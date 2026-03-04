# model_lgbm.py
# LightGBM binary classifier for thunderstorm end prediction.

import logging
from typing import Optional, Tuple

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from config import CFG

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    Wraps LightGBM with early stopping, feature importance, and SHAP support.
    """

    def __init__(self):
        cfg = CFG.lgbm
        self.model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )
        self.feature_names = CFG.features.feature_names
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMModel":
        """
        Fit the model. If validation data is provided, uses early stopping.
        """
        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            feature_name=self.feature_names,
            callbacks=callbacks,
        )

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "average_precision"
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=CFG.lgbm.early_stopping_rounds,
                    verbose=True,
                )
            )

        logger.info("Training LightGBM...")
        self.model.fit(**fit_kwargs)
        self.is_fitted = True

        # Log training metrics
        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
            auc = roc_auc_score(y_val, val_proba)
            ap = average_precision_score(y_val, val_proba)
            logger.info(f"LightGBM val — ROC-AUC: {auc:.4f} | AvgPrecision: {ap:.4f}")

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 (storm ended)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    def feature_importance(self, importance_type: str = "gain") -> dict:
        """Return feature importances as a sorted dict."""
        imp = self.model.booster_.feature_importance(importance_type=importance_type)
        names = self.model.booster_.feature_name()
        return dict(sorted(zip(names, imp), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for interpretability.
        Returns array of shape (n_samples, n_features).
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            # For binary classification, shap_values is a list [class0, class1]
            if isinstance(shap_values, list):
                return shap_values[1]
            return shap_values
        except ImportError:
            logger.warning("shap not installed. Run: pip install shap")
            return np.zeros_like(X)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"LightGBM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        model = joblib.load(path)
        logger.info(f"LightGBM model loaded from {path}")
        return model
