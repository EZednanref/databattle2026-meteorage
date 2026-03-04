![](./gif.gif)

# ⛈️ Thunderstorm End Prediction Model

Predicts the probability that a thunderstorm has ended near an airport, updated after every lightning strike.

## Project Structure

```
storm_model/
├── config.py               # All hyperparameters and settings
├── data_loader.py          # Data loading and validation
├── features.py             # Feature engineering pipeline
├── model_lgbm.py           # LightGBM model
├── model_lstm.py           # LSTM model (PyTorch)
├── model_ensemble.py       # Ensemble + probability calibration
├── train.py                # Full training pipeline
├── evaluate.py             # Evaluation metrics and plots
├── predict.py              # Real-time inference
├── requirements.txt        # Dependencies
└── README.md
```

**2026-03-04 11:44:45,221 [INFO] **main** — [Biarritz] Test results — ROC-AUC: 0.877 | AvgPrecision: 0.301 | Brier: 0.0487 | Mean CI width: 0.695
2026-03-04 11:42:31,423 [INFO] **main** — [Pise] Test results — ROC-AUC: 0.856 | AvgPrecision: 0.071 | Brier: 0.0196 | Mean CI width: 0.227
2026-03-04 11:37:21,237 [INFO] **main** — [Nantes] Test results — ROC-AUC: 0.784 | AvgPrecision: 0.184 | Brier: 0.0331 | Mean CI width: 0.323
2026-03-04 11:36:01,208 [INFO] **main** — [Ajaccio] Test results — ROC-AUC: 0.898 | AvgPrecision: 0.327 | Brier: 0.0504 | Mean CI width: 0.357
2026-03-04 11:33:31,478 [INFO] **main** — [Bastia] Test results — ROC-AUC: 0.914 | AvgPrecision: 0.241 | Brier: 0.0235 | Mean CI width: 0.184
**

**Bastia & Ajaccio** are our best models — high ROC-AUC, well-calibrated, reasonable CI width.

**Biarritz** has a CI width of 0.695 which is very wide — the model is uncertain on almost every prediction. This could mean the 2020 validation year was unusual for Biarritz (atypical storm season).

**Pise** has a very low Average Precision of 0.071 — barely better than random. This is likely the recording system issue mentioned in the data description (different system in 2016). Even though we excluded 2016 IC strikes, the overall data quality for Pise may be lower.

**Nantes** at 0.784 ROC-AUC is the weakest overall. With only 4,378 strikes total it has the least training data of all airports — the model simply hasn't seen enough storms to generalize well.

**The good news** is that all 5 airports are well above random on ROC-AUC, and all Brier scores are well below 0.10, meaning probability outputs are trustworthy for operational use.

## Quick Start

```bash
pip install -r requirements.txt

# Train the full model
python train.py --data_path data/lightning.parquet --airport Bron

# Evaluate on test set
python evaluate.py --model_path models/ensemble_Bron.pkl

# Real-time prediction
python predict.py --model_path models/ensemble_Bron.pkl --alert_id 42
```

## Output Format

```
Time (UTC) | P(ended) | 90% CI        | Advisory
-----------|----------|---------------|---------------------------
12:00      |   80%    | [73% – 87%]   | 🟡 Storm possibly ending
12:10      |   94%    | [89% – 97%]   | 🟢 Storm likely ended
```

## Airports Supported

* Bastia (BIA): 42.5527°N, 9.4837°E
* Ajaccio (AJA): 41.9236°N, 8.8029°E
* Nantes (NTE): 47.1532°N, -1.6107°E
* Pisa (PSA): 43.695°N, 10.399°E
* Biarritz (BIQ): 43.4683°N, -1.524°E

## Notes

- Pisa 2016 intra-cloud data is excluded by default (different recording system)
- All features use only backward-looking windows (no data leakage)
- Train/test split is temporal (train ≤ 2021, val = 2022, test ≥ 2023)
