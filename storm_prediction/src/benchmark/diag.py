import pandas as pd

# Distribution de la durée des orages par année
# df = pd.read_csv("../../data/processed/storms.csv", parse_dates=["start_time"])
# df["year"] = df["start_time"].dt.year
# print(df.groupby("year")["duration_min"].describe()[["mean","count"]])


# df = pd.read_csv("../../data/processed/features.csv", parse_dates=["snapshot_time"])
# df = df.sort_values("snapshot_time").dropna()
#
# n = len(df)
# df["split"] = "train"
# df.iloc[int(n*0.85):, df.columns.get_loc("split")] = "test"
# df.iloc[int(n*0.70):int(n*0.85), df.columns.get_loc("split")] = "val"
#
# print(df.groupby("split")["snapshot_time"].agg(["min", "max"]))
# print(df.groupby("split")["label_binary"].mean())
#
# df = pd.read_csv("../../data/processed/storms.csv", parse_dates=["start_time"])
# df["year"] = df["start_time"].dt.year
# print(df.groupby("year")["n_lightnings"].describe()[["mean","count"]])
df = pd.read_csv("../../data/processed/features.csv", parse_dates=["snapshot_time"])
df["year"] = df["snapshot_time"].dt.year
print(df.groupby("year")[["global_icloud_ratio", "storm_age_min", "w20_lightning_rate"]].mean())
