"""
Optimisation LightGBM avec Optuna sur la prédiction de fin d'orage.
Split temporel strict — pas de leakage.

Input  : data/processed/features.csv
Output : output/lgbm/metrics.csv + plots + best_params.json

from : https://optuna.readthedocs.io/en/stable/tutorials/10_key_features/003_efficient_optimization_algorithms.html
from : https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier

import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH  = "../../data/processed/features.csv"
OUTPUT_DIR     = "output/lgbm"
N_TRIALS       = 50      # nombre d'essais Optuna
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15

EXCLUDE_COLS   = [
    "storm_id", "airport", "snapshot_time",
    "label_binary", "time_to_end_min", "event",
    "time_since_last",
    "storm_duration_min",
]
DROP_LOW_SIGNAL = [
    "w5_dist_mean", "w10_dist_mean", "w20_dist_mean",
    "w5_loc_error_mean", "w10_loc_error_mean", "w20_loc_error_mean",
    "w5_loc_error_trend", "w10_loc_error_trend", "w20_loc_error_trend",
]
# ─────────────────────────────────────────────────────────────────────────────


# ── Chargement & split ────────────────────────────────────────────────────────

def load_and_split(path: str):
    df = pd.read_csv(path, parse_dates=["snapshot_time"])
    df = df.sort_values("snapshot_time").reset_index(drop=True)
    df = df.dropna()
    # Dans load_and_split, après le dropna() :
    df = df[df["snapshot_time"].dt.year >= 2017]
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and c not in DROP_LOW_SIGNAL]

    X = df[feature_cols].values
    y = df["label_binary"].values
    feature_names = feature_cols

    n        = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": (X[:train_end],       y[:train_end]),
        "val":   (X[train_end:val_end], y[train_end:val_end]),
        "test":  (X[val_end:],          y[val_end:]),
    }

    for name, (Xs, ys) in splits.items():
        print(f"  {name:5s} : {len(Xs):,} lignes — label=1 : {ys.mean():.1%}")

    return splits, feature_names


def make_objective(splits, feature_names):
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]

    def objective(trial):
        # from : https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        model = LGBMClassifier(
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 20, 200),
            max_depth         = trial.suggest_int("max_depth", 3, 12),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            random_state      = 42,
            verbose           = -1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    return objective


def train_final(best_params, splits, feature_names):
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
    model.fit(X_trainval, y_trainval)
    return model


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(proba, y_test) -> dict:
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    mask = precision >= 0.95
    recall_95 = round(float(recall[mask].max()) if mask.any() else 0.0, 4)

    return {
        "AUC":         round(roc_auc_score(y_test, proba), 4),
        "Brier":       round(brier_score_loss(y_test, proba), 4),
        "Recall@95p":  recall_95,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(model, proba, splits, feature_names):
    X_test, y_test = splits["test"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    RocCurveDisplay.from_predictions(y_test, proba, name="LightGBM", ax=axes[0])
    axes[0].set_title("Courbe ROC — LightGBM")

    # Calibration
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10,
                                        name="LightGBM", ax=axes[1])
    axes[1].set_title("Calibration — LightGBM")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_calibration.png"), dpi=150)
    plt.close()
    print("  → roc_calibration.png")

    # Distribution des probas
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba[y_test == 0], bins=50, alpha=0.6, label="label=0", color="steelblue", density=True)
    ax.hist(proba[y_test == 1], bins=50, alpha=0.6, label="label=1", color="tomato",    density=True)
    ax.set_xlabel("Probabilité prédite")
    ax.set_title("Distribution des probabilités par classe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "proba_distribution.png"), dpi=150)
    plt.close()
    print("  → proba_distribution.png")

    # SHAP
    # from : https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Python%20Version%20of%20Tree%20SHAP.html
    X_train, _ = splits["train"]
    sample_idx = np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)
    sample     = X_train[sample_idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  → shap_summary.png")

    # Optuna optimization history
    return


def plot_optuna_history(study):
    # from : https://optuna.readthedocs.io/en/stable/reference/visualization/index.html
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trials_df = study.trials_dataframe()
    axes[0].plot(trials_df["number"], trials_df["value"], alpha=0.5, marker="o", ms=3)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("AUC (val)")
    axes[0].set_title("Historique Optuna")

    best_so_far = trials_df["value"].cummax()
    axes[1].plot(trials_df["number"], best_so_far, color="tomato")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Meilleur AUC")
    axes[1].set_title("Convergence")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optuna_history.png"), dpi=150)
    plt.close()
    print("  → optuna_history.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Chargement & split...")
    splits, feature_names = load_and_split(FEATURES_PATH)

    print(f"\nOptimisation Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(splits, feature_names), n_trials=N_TRIALS,
                   show_progress_bar=True)
    print(f"Meilleur AUC val : {study.best_value:.4f}")
    print(f"Meilleurs params : {study.best_params}")
    

    # Sauvegarde des params
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("\nEntraînement final (train+val)...")
    model = train_final(study.best_params, splits, feature_names)

    X_test, y_test = splits["test"]
    proba = model.predict(X_test)

    m = evaluate(proba, y_test)
    print(f"\n{'='*40}")
    print(f"AUC        : {m['AUC']}")
    print(f"Brier      : {m['Brier']}")
    print(f"Recall@95p : {m['Recall@95p']}")

    pd.DataFrame([{"model": "LightGBM_Optuna", **m}]).to_csv(
        os.path.join(OUTPUT_DIR, "metrics.csv"), index=False
    )

    print("\nPlots...")
    plot_results(model, proba, splits, feature_names)
    plot_optuna_history(study)

    print(f"\nRésultats → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
