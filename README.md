# Data Battle IA PAU 2026

## Equipe
- Nom de l'équipe : Zango le dozo
- Membres :
    - Kraus Antoine
    - Oszust Jordi
    - Champeyrol Matys
    - Fritsch Lucas
    - Malherbe Axel
    - Fernandez Enzo
    
## Problématique
Prédire la fin d’un orage 30 km autour d’un aéroport.
Prédire la trajectoire, la localisation, l’heure.


## Solution proposée

## Stack technique

### Langage de Programmation
- **Python** (langage principal)

### Frameworks et Bibliothèques de Base
- **NumPy** (≥ 1.24) — Calculs numériques
- **Pandas** (≥ 2.0) — Manipulation et analyse de données
- **SciPy** (≥ 1.10) — Calculs scientifiques
- **scikit-learn** (≥ 1.3) — Utilitaires ML (prétraitement, métriques, calibration)

### Frameworks Machine Learning et Deep Learning
- **PyTorch** (≥ 2.0) — Réseaux de neurones LSTM
- **LightGBM** (≥ 4.0) — Classifieur par boosting de gradient (modèle principal)
- **XGBoost** — Boosting de gradient (analyse de features et benchmarking)
- **Optuna** — Optimisation d'hyperparamètres (recherche bayésienne)

### Composants ML Avancés
- **SHAP** (≥ 0.43) — Explicabilité des modèles et importance des features
- **MAPIE** (≥ 0.7) — Intervalles de prédiction conformaux pour quantification d'incertitude
- **lifelines** (≥ 0.27) — Analyse de survie (modèles Weibull AFT, Cox PH)
- **imbalanced-learn** — SMOTE pour gérer le déséquilibre de classes dans les splits temporels

### Traitement de Données et I/O
- **PyArrow** (≥ 12.0) — Support des fichiers Parquet
- **fastparquet** (≥ 2023.4) — Moteur alternatif Parquet
- **joblib** (≥ 1.3) — Traitement parallèle et sérialisation de modèles
- **tqdm** — Barres de progression pour les boucles de traitement

### Visualisation et Analyse
- **Matplotlib** (≥ 3.7) — Graphiques et visualisations d'évaluation
- **Seaborn** (≥ 0.12) — Visualisation statistique des données
- HTML/JavaScript — Cartes interactives personnalisées pour l'analyse de direction des orages

### Environnements de Développement
- **Jupyter Notebooks** — Analyse interactive et pipelines (ex: run_pipeline.ipynb, pipeline.ipynb)
- **Git** — Contrôle de version

### Architecture du Projet
- **Dossier LSTM** : Ensemble deep learning (LSTM + LightGBM avec calibration de probabilités)
- **Dossier storm_prediction** : Approche multi-facettes (benchmarking, analyse de survie, analyse de features, prédiction de trajectoire temporelle)
- Workflow principal : Prétraitement des données → Ingénierie des features → Entraînement des modèles → Évaluation → Inférence en temps réel

Toutes les dépendances sont spécifiées dans [lstm/requirements.txt](lstm/requirements.txt). Le module storm_prediction utilise des bibliothèques supplémentaires (XGBoost, Optuna, lifelines, imbalanced-learn) non listées là mais intégrées dans ses modules Python.


## Installation et execution

### Prérequis

#### Cloner le dépôt
```bash
git clone https://github.com/EZednanref/databattle2026-meteorage.git
cd databattle2026-meteorage
```

#### Créer un environnement virtuel
```bash
python -m venv env
source env/bin/activate  # Sur Windows: env\Scripts\activate
```

#### Installer les dépendances
```bash
pip install -r lstm/requirements.txt
```

### Execution


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

