# evaluate.py
# Full evaluation suite: calibration, discrimination, lead time analysis,
# per-airport breakdown, and publication-ready plots.
#
# Usage:
#   python evaluate.py --model_path models/ensemble_Bron.pkl \
#                      --data_path data/lightning.parquet \
#                      --airport Bron

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve

from config import CFG, AIRPORTS
from data_loader import load_data, split_temporal
from features import build_features, build_sequences, load_scaler
from model_ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(model_path: str, data_path: str, airport: str,
             output_dir: str = "outputs") -> dict:
    """
    Run the full evaluation suite and save plots.

    Returns
    -------
    metrics : dict
        Summary of all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model ----
    ensemble = EnsembleModel.load(model_path)
    scaler_path = os.path.join(CFG.paths.model_dir, f"scaler_{airport}.pkl")
    scaler = load_scaler(scaler_path)

    # ---- Load test data ----
    df = load_data(data_path, airport=airport)
    _, _, test_df = split_temporal(df)

    test_feat, X_test, y_test, _ = build_features(test_df, scaler=scaler)
    X_test_seq, y_test_seq, test_seq_idx = build_sequences(test_feat, X_test)
    X_test_flat = X_test[test_seq_idx]
    y_test_flat = y_test[test_seq_idx]

    # Align feature_df with sequence rows
    test_feat_aligned = test_feat.iloc[test_seq_idx].reset_index(drop=True)
    test_feat_aligned["pred_proba"] = None
    test_feat_aligned["pred_lower"] = None
    test_feat_aligned["pred_upper"] = None

    # ---- Get predictions ----
    proba, lower, upper = ensemble.predict(X_test_flat, X_test_seq)
    test_feat_aligned["pred_proba"] = proba
    test_feat_aligned["pred_lower"] = lower
    test_feat_aligned["pred_upper"] = upper

    # ---- Compute all metrics ----
    metrics = {}

    # 1. Calibration
    metrics["brier_score"] = brier_score_loss(y_test_flat, proba)

    # 2. Discrimination
    metrics["roc_auc"] = roc_auc_score(y_test_flat, proba)
    metrics["average_precision"] = average_precision_score(y_test_flat, proba)

    # 3. Lead time
    lead_stats = compute_lead_time_stats(test_feat_aligned, threshold=CFG.thresholds.allclear_threshold)
    metrics.update(lead_stats)

    # 4. Confusion matrix at operational threshold
    y_pred = (proba >= CFG.thresholds.allclear_threshold).astype(int)
    cm = confusion_matrix(y_test_flat, y_pred)
    metrics["false_allclear_rate"] = cm[0, 1] / (cm[0, 0] + cm[0, 1] + 1e-9)
    metrics["missed_end_rate"] = cm[1, 0] / (cm[1, 0] + cm[1, 1] + 1e-9)

    _print_metrics(metrics, airport)

    # ---- Plots ----
    fig = _make_evaluation_figure(
        y_true=y_test_flat,
        proba=proba,
        test_feat=test_feat_aligned,
        airport=airport,
    )
    plot_path = os.path.join(output_dir, f"evaluation_{airport}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Evaluation plot saved to {plot_path}")

    # ---- Save metrics to CSV ----
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_{airport}.csv"), index=False)

    return metrics


# ---------------------------------------------------------------------------
# Lead time analysis
# ---------------------------------------------------------------------------

def compute_lead_time_stats(test_feat: pd.DataFrame, threshold: float) -> dict:
    """
    For each alert, find how many minutes before the storm actually ended
    the model first exceeded `threshold`.
    """
    lead_times = []
    false_early = 0
    missed = 0

    for alert_id, group in test_feat.groupby("airport_alert_id"):
        group = group.sort_values("date")

        # Ground truth: timestamp of last CG strike
        last_cg_rows = group[group["is_last_lightning_cloud_ground"] == True]
        if last_cg_rows.empty:
            continue
        true_end_time = last_cg_rows["date"].iloc[0]

        # First time model exceeds threshold
        triggered = group[group["pred_proba"] >= threshold]
        if triggered.empty:
            missed += 1
            continue

        first_trigger = triggered["date"].iloc[0]
        lead_min = (true_end_time - first_trigger).total_seconds() / 60
        lead_times.append(lead_min)

        if lead_min < 0:
            false_early += 1  # model triggered before storm actually ended

    s = pd.Series(lead_times)
    n = len(s)
    return {
        "n_alerts_evaluated": n,
        "lead_time_median_min": s.median() if n > 0 else np.nan,
        "lead_time_p25_min": s.quantile(0.25) if n > 0 else np.nan,
        "lead_time_p75_min": s.quantile(0.75) if n > 0 else np.nan,
        "pct_lead_gt_10min": (s > 10).mean() if n > 0 else np.nan,
        "pct_lead_gt_20min": (s > 20).mean() if n > 0 else np.nan,
        "pct_false_early": false_early / n if n > 0 else np.nan,
        "n_missed_alerts": missed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_evaluation_figure(y_true, proba, test_feat, airport):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Model Evaluation — {airport}", fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Calibration curve
    ax1 = fig.add_subplot(gs[0, 0])
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
    ax1.plot(prob_pred, prob_true, "s-", color="steelblue", label="Model")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # 2. ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax2, name="Ensemble")
    ax2.set_title("ROC Curve")
    ax2.plot([0, 1], [0, 1], "--", color="gray")

    # 3. Precision-Recall curve
    ax3 = fig.add_subplot(gs[0, 2])
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax3, name="Ensemble")
    ax3.set_title("Precision-Recall Curve")

    # 4. Lead time distribution
    ax4 = fig.add_subplot(gs[1, 0])
    lead_stats = compute_lead_time_stats(test_feat, threshold=CFG.thresholds.allclear_threshold)
    # Re-compute raw lead times for histogram
    lead_times = _get_lead_times(test_feat, CFG.thresholds.allclear_threshold)
    if lead_times:
        ax4.hist(lead_times, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax4.axvline(0, color="red", linestyle="--", label="Storm end")
        ax4.axvline(np.median(lead_times), color="orange", linestyle="-",
                    label=f"Median: {np.median(lead_times):.1f} min")
        ax4.set_xlabel("Lead time (minutes before storm end)")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Lead Time Distribution\n(threshold={CFG.thresholds.allclear_threshold:.0%})")
        ax4.legend()

    # 5. Probability distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(proba[y_true == 0], bins=40, alpha=0.6, color="steelblue", label="Active storms")
    ax5.hist(proba[y_true == 1], bins=40, alpha=0.6, color="tomato", label="Storm end")
    ax5.axvline(CFG.thresholds.allclear_threshold, color="black", linestyle="--",
                label=f"Threshold={CFG.thresholds.allclear_threshold:.0%}")
    ax5.set_xlabel("Predicted probability P(storm ended)")
    ax5.set_ylabel("Count")
    ax5.set_title("Score Distribution")
    ax5.legend()

    # 6. Threshold sensitivity
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(0.5, 0.99, 50)
    false_early_rates, missed_rates, lead_medians = [], [], []
    for t in thresholds:
        stats = compute_lead_time_stats(test_feat, threshold=t)
        false_early_rates.append(stats["pct_false_early"])
        missed_rates.append(stats["n_missed_alerts"])
        lead_medians.append(stats["lead_time_median_min"])
    ax6.plot(thresholds, false_early_rates, label="False early-call rate", color="tomato")
    ax6_r = ax6.twinx()
    ax6_r.plot(thresholds, lead_medians, label="Median lead time (min)", color="steelblue", linestyle="--")
    ax6.axvline(CFG.thresholds.allclear_threshold, color="black", linestyle=":", alpha=0.7)
    ax6.set_xlabel("Decision threshold")
    ax6.set_ylabel("False early-call rate", color="tomato")
    ax6_r.set_ylabel("Median lead time (min)", color="steelblue")
    ax6.set_title("Threshold Sensitivity")
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_r.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    return fig


def _get_lead_times(test_feat, threshold):
    lead_times = []
    for _, group in test_feat.groupby("airport_alert_id"):
        group = group.sort_values("date")
        last_cg = group[group["is_last_lightning_cloud_ground"] == True]
        if last_cg.empty:
            continue
        true_end = last_cg["date"].iloc[0]
        triggered = group[group["pred_proba"] >= threshold]
        if triggered.empty:
            continue
        lead = (true_end - triggered["date"].iloc[0]).total_seconds() / 60
        lead_times.append(lead)
    return lead_times


def _print_metrics(metrics, airport):
    logger.info(f"\n{'=' * 50}")
    logger.info(f"EVALUATION RESULTS — {airport}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Brier Score:          {metrics['brier_score']:.4f}  (target < 0.10)")
    logger.info(f"  ROC-AUC:              {metrics['roc_auc']:.3f}  (target > 0.85)")
    logger.info(f"  Average Precision:    {metrics['average_precision']:.3f}  (target > 0.70)")
    logger.info(f"  Median lead time:     {metrics['lead_time_median_min']:.1f} min  (target > 10)")
    logger.info(f"  % lead > 10 min:      {metrics['pct_lead_gt_10min']:.1%}")
    logger.info(f"  % lead > 20 min:      {metrics['pct_lead_gt_20min']:.1%}")
    logger.info(f"  False early-call:     {metrics['pct_false_early']:.1%}  (target < 5%)")
    logger.info(f"  False all-clear rate: {metrics['false_allclear_rate']:.1%}  (target < 5%)")
    logger.info(f"  Missed alert ends:    {metrics['n_missed_alerts']}  (target < 2%)")
    logger.info(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate thunderstorm end prediction model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--airport", type=str, required=True, choices=list(AIRPORTS.keys()))
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.airport, args.output_dir)


if __name__ == "__main__":
    main()
