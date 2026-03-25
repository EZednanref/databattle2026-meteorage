import pandas as pd
"""
Il faut encore peaufiner, pour genre chaque villes 
"""
# --- CONFIG ---
METEO_FILE = "meteo.csv"
LIGHTNING_FILE = "data.csv"
OUTPUT_FILE = "enrichies.csv"
SEP_METEO = ";"
SEP_LIGHTNING = ";"

# --- LOAD ---
meteo = pd.read_csv(METEO_FILE, sep=SEP_METEO, low_memory=False)
lightning = pd.read_csv(LIGHTNING_FILE, sep=SEP_LIGHTNING, low_memory=False)

# --- NORMALIZE KEYS ---
# from: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
meteo["__hour_key"] = pd.to_datetime(meteo["AAAAMMJJHH"], format="%Y%m%d%H")
lightning["__hour_key"] = pd.to_datetime(lightning["date"], utc=True).dt.tz_localize(None).dt.floor("h")

meteo["__city_key"] = meteo["NOM_USUEL"].str.strip().str.upper()
lightning["__city_key"] = lightning["airport"].str.strip().str.upper()

# --- MERGE avec suffixes pour distinguer old/new ---
# from: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
meteo_cols = [c for c in meteo.columns if c not in ("__hour_key", "__city_key")]
shared_cols = [c for c in meteo_cols if c in lightning.columns]

merged = lightning.merge(
    meteo,
    on=["__city_key", "__hour_key"],
    how="left",
    suffixes=("", "__new")
)

# --- FILLNA : remplir les NaN existants avec les nouvelles valeurs ---
# from: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.combine_first.html
for col in shared_cols:
    new_col = f"{col}__new"
    if new_col in merged.columns:
        merged[col] = merged[col].fillna(merged[new_col])
        merged.drop(columns=[new_col], inplace=True)

# Ajouter les colonnes météo absentes du lightning original
for col in meteo_cols:
    if col not in merged.columns:
        merged[col] = merged.get(f"{col}__new")

# --- CLEANUP ---
merged.drop(columns=["__city_key", "__hour_key"], inplace=True)

# --- EXPORT ---
merged.to_csv(OUTPUT_FILE, index=False, sep=";")
print(f"Done — {len(merged)} rows written to {OUTPUT_FILE}")
unmatched = merged["NOM_USUEL"].isna().sum()
print(f"Lignes toujours sans correspondance météo : {unmatched}")
