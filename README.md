![](./gif.gif)



# AVANT TOUT

Va falloir faire un croisement de données.
C'est à dire trouver des données open source à propos de la météo autour d'Ajaccio, Bastia, Biarritz, Nantes, Pise.

Faire un script permettant d'enrichir les données en les croisant.

A l'aide de ces données enrichies, faire un benchmark de features, afin de cerner les plus impactantes pour la prédiction de fin d'orage.

Sources possibles : 

- https://meteo.data.gouv.fr/datasets?category=climatologique-base
- https://meteo.data.gouv.fr/datasets?category=climatologique-changement-climatique
- https://meteo.data.gouv.fr/datasets?category=observations






# Benchmark — Prédiction de fin d'orage (Étape 1)

## Objectif

À partir d'un historique d'éclairs sur un aéroport, prédire **la borne supérieure de fin d'orage** (ex: "l'orage sera terminé avant 17h30 avec 95% de confiance").

Deux formulations sont benchmarkées en parallèle :

| Formulation | Sortie | Modèles |
|---|---|---|
| **Binaire** | "L'orage se termine-t-il dans les X prochaines minutes ?" | LR, XGBoost, LightGBM, LSTM |
| **Survival** | "Quelle est la probabilité que l'orage dure encore T minutes ?" | Weibull AFT, Cox PH |

---

## Définition des labels

**Règle de segmentation :** un orage = séquence d'éclairs sur un aéroport sans gap > 30 min.

```
storm_id  start_time  end_time  duration_min  [features agrégées]
```

**Label binaire** (à paramétrer) : `1` si `duration_from_now <= T`, `0` sinon.  
**Label survival** : `(duration_min, event=1)` — l'événement est toujours observé ici.

---

## Feature Engineering

Calculées par **fenêtre glissante** sur les N derniers éclairs de l'orage en cours :

### Temporelles
- `time_since_last_lightning` — signal clé pour la fin
- `inter_lightning_gap_mean / max` sur fenêtre 5, 10, 20 éclairs
- `lightning_rate_trend` — pente de la fréquence (décélération = signal fort)
- `storm_age_min` — durée depuis le premier éclair

### Spatiales
- `dist_mean / dist_trend` — éloignement progressif du centre ?
- `azimuth_std` — dispersion angulaire
- `maxis_mean / trend` — évolution de la taille de la cellule orageuse

### Physiques
- `amplitude_mean / std / trend` — intensité et tendance
- `icloud_ratio` — proportion de foudre nuage-nuage (↑ en fin d'orage)
- `cloud_ground_ratio` — proportion nuage-sol

---

## Structure du projet

```
storm_end_prediction/
│
├── data/
│   ├── raw/data.csv
│   └── processed/
│       ├── storms.csv          # segments d'orages labelisés
│       └── features.csv        # features par fenêtre glissante
│
├── src/
│   ├── preprocessing/
│   │   ├── segment_storms.py   # règle 30 min → storm_id
│   │   └── build_features.py   # fenêtres glissantes → feature matrix
│   │
│   ├── models/
│   │   ├── baseline_lr.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_model.py
│   │   └── survival_model.py   # lifelines : Weibull AFT + Cox PH
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # AUC, Brier score, calibration curve
│   │   └── benchmark.py        # runner comparatif
│   │
│   └── config.py               # GAP_MINUTES=30, WINDOW_SIZES, T_horizon, etc.
│
├── notebooks/
│   └── 01_EDA.ipynb
│
└── README.md
```

---

## Métriques d'évaluation

### Modèles binaires / classifieurs
- **AUC-ROC** — discrimination générale
- **Brier Score** — qualité probabiliste (calibration)
- **Calibration curve** — essentiel si la sortie est une probabilité utilisée en prod
- **Recall à 95%** — métrique métier : "couvrir 95% des fins d'orage réelles"

### Modèles survival
- **C-index (Concordance)** — équivalent AUC pour la survival
- **Integrated Brier Score**
- **Survival curves** à t=30, 60, 90 min

---

## Split temporel

> Ne pas faire de split aléatoire — risque de data leakage temporel.

```
Train : orages avant date D
Val   : orages entre D et D+30j
Test  : orages après D+30j
```

Ou **Walk-Forward Validation** si peu de données.

---

## Dépendances probables

```
pandas, numpy
scikit-learn
xgboost, lightgbm
torch (LSTM)
lifelines (survival)
matplotlib, shap
```

---

## Décision sur la formulation

Recommandation : **commencer par la formulation survival** (Weibull AFT).

- Modélise naturellement "le temps jusqu'à la fin"
- Gère les orages encore en cours (données censurées) si besoin
- Donne directement un percentile 95% → borne supérieure pour l'étape 2
- Plus interprétable que LSTM pour un premier benchmark

La formulation binaire reste utile comme **baseline rapide** et pour comparer la calibration.

