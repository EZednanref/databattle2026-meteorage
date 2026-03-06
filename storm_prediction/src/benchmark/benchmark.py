import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import roc_auc_score, brier_score_loss
from sklearn.metrics         import RocCurveDisplay
from sklearn.calibration     import CalibrationDisplay

from xgboost import XGBClassifier
import shap


FEATURES_PATH = "../../data/processed/features.csv"
OUTPUT_DIR    = "output/benchmark"

EXCLUDE_COLS  = [
    "storm_id", "airport", "snapshot_time",
    "label_binary", "time_to_end_min", "event",
    "time_since_last",   # toujours ~0 au moment du snapshot
    "storm_duration_min",
]

DROP_LOW_SIGNAL = [
    "w5_dist_mean", "w10_dist_mean", "w20_dist_mean",
    "w5_loc_error_mean", "w10_loc_error_mean", "w20_loc_error_mean",
    "w5_loc_error_trend", "w10_loc_error_trend", "w20_loc_error_trend",
]

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15

def load_and_split(path: str):
    df = pd.read_csv(path, parse_dates=["snapshot_time"])
    df = df.sort_values("snapshot_time").reset_index(drop=True)
    df = df.dropna()

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and c not in DROP_LOW_SIGNAL]

    X = df[feature_cols]
    y = df["label_binary"]

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": (X.iloc[:train_end],  y.iloc[:train_end]),
        "val":   (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        "test":  (X.iloc[val_end:],    y.iloc[val_end:]),
    }

    print(f"  Features : {len(feature_cols)}")
    for name, (Xs, ys) in splits.items():
        print(f"  {name:5s} : {len(Xs):,} lignes — label=1 : {ys.mean():.1%}")

    return splits, feature_cols


def build_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.1,
                random_state=42
            ))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=20,
            random_state=42,
            verbosity=0,
        )
    }


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "AUC":         round(roc_auc_score(y_test, proba), 4),
        "Brier":       round(brier_score_loss(y_test, proba), 4),
        "Recall@95p":  recall_at_precision(y_test, proba, target_precision=0.95),
    }


def recall_at_precision(y_true, proba, target_precision: float) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, proba)
    mask = precision >= target_precision
    return round(float(recall[mask].max()) if mask.any() else 0.0, 4)


def plot_roc_and_calibration(results: dict, splits: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X_test, y_test = splits["test"]

    for name, (model, _) in results.items():
        proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        CalibrationDisplay.from_predictions(y_test, proba, n_bins=10,
                                            name=name, ax=axes[1])

    axes[0].set_title("Courbe ROC")
    axes[1].set_title("Calibration")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_calibration.png"), dpi=150)
    plt.close()
    print("  → roc_calibration.png")


def plot_shap(model, X_train, feature_cols: list):
    explainer  = shap.TreeExplainer(model)
    sample     = X_train.sample(min(2000, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  → shap_summary.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splits, feature_cols = load_and_split(FEATURES_PATH)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    models   = build_models()
    results  = {}
    metrics  = []

    for name, model in models.items():
        print(f"\nEntraînement : {name}")

        if name == "XGBoost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        m = evaluate(model, X_test, y_test)
        results[name] = (model, m)
        metrics.append({"model": name, **m})
        print(f"  AUC={m['AUC']}  Brier={m['Brier']}  Recall@95p={m['Recall@95p']}")

    plot_roc_and_calibration(results, splits)
    xgb_model = results["XGBoost"][0]
    plot_shap(xgb_model, X_train, feature_cols)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))
    print(f"\nRésultats → {OUTPUT_DIR}/")
    
    xgb = results["XGBoost"][0]
    proba = xgb.predict_proba(X_test)[:, 1]
    print(pd.Series(proba).describe())
    print("% proba > 0.8 :", (proba > 0.8).mean())

if __name__ == "__main__":
    main()
