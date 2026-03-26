import pickle
import numpy as np
import pandas as pd

from train_temporal_trajectory import build_features, META_COLS
from segment_storm import assign_storm_ids

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "../model_output/lgbm_model.pkl"
TEST_CSV    = "dataset_set.csv"
OUTPUT_PATH = "../model_output/predictions.csv"

MIN_LIGHTNINGS = 3

def main():
    print("Chargement du modèle...")
    with open(MODEL_PATH, "rb") as f:
        lgbm = pickle.load(f)

    print("Chargement des données...")
    df = pd.read_csv(TEST_CSV, parse_dates=["date"])

    print("Segmentation en orages...")
    df = assign_storm_ids(df, gap_minutes=30)
    print(f"  {df['storm_id'].nunique()} orages identifiés")

    storm_sizes = df.groupby("storm_id").size()
    df = df[df["storm_id"].isin(storm_sizes[storm_sizes >= MIN_LIGHTNINGS].index)]

    print("Construction des features...")
    features = build_features(df)

    exclude = set(META_COLS + ["is_last_lightning_cloud_ground"])
    feature_cols = [c for c in features.columns
                    if c not in exclude
                    and not c.startswith("Q")
                    and not c.endswith("_first")]

    X = features[feature_cols].select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    # Aligne les colonnes sur celles vues à l'entraînement
    # from: scikit-learn.org/stable/modules/model_persistence.html
    X = X.reindex(columns=lgbm.feature_name_, fill_value=0)

    print("Inférence...")
    features["proba_cloud_ground"] = lgbm.predict_proba(X)[:, 1]
    features["prediction"]         = lgbm.predict(X)

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"→ {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
