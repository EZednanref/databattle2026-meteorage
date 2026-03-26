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
Pipeline temps réels de tracking d’orage + visualisation web interactive (storm_prediction/run_pipeline.ipynb)
Modèle LightGBM de prédiction de fin d’orage.

## Stack technique

### Langage de Programmation
- **Python** (langage principal)

### Frameworks et Bibliothèques de Base
- **NumPy** (≥ 1.24) — Calculs numériques
- **Pandas** (≥ 2.0) — Manipulation et analyse de données
- **SciPy** (≥ 1.10) — Calculs scientifiques
- **scikit-learn** (≥ 1.3) — Utilitaires ML (prétraitement, métriques, calibration)

### Frameworks Machine Learning et Deep Learning
- **PyTorch** (≥ 2.0) 
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
- **Dossier DEV** : Ensemble deep learning (LightGBM avec calibration de probabilités)
- **Dossier storm_prediction** : Approche multi-facettes (benchmarking, analyse de survie, analyse de features, prédiction de trajectoire temporelle)
- Workflow principal : Prétraitement des données → Ingénierie des features → Entraînement des modèles → Évaluation → Inférence en temps réel

Toutes les dépendances sont spécifiées dans [requirements.txt](requirements.txt). Le module storm_prediction utilise des bibliothèques supplémentaires (XGBoost, Optuna, lifelines, imbalanced-learn) non listées là mais intégrées dans ses modules Python.


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
pip install -r requirements.txt
```

### Execution
- Interface web intéractive de tracking d'orage : 
Simplement éxecuter le note book :

```
storm_prediction/run_pipeline.ipynb**
```

- Lancer l'algorithme de prédiction de fin d'orage :

```bash
python dev/final_folder/predict.py
```

## Structure du projet

```
storm_prediction/
│
├── data/
│   ├── raw/
│   │   ├── data_with_storm_id.csv
│   │   └── data.csv
│   └── final_folder/
│       ├── pipeline.ipynb
│       ├── requirements.txt
│       ├── feature_analysis/
│       │   ├── feature_ranking.csv
│       │   └── optimal_features.py
│       ├── inference/
│       │   ├── dataset_set.csv
│       │   ├── predict.py
│       │   ├── segment_storm.py
│       │   ├── storms.csv
│       │   └── train_temporal_trajectory.py
│       ├── model_output/
│       │   └── rapport_lgbm_trajectory.txt
│       ├── preprocessing/
│       │   └── segment_storm.py
│       └── training/
│           ├── data_enrichie_features.csv
│           ├── lgbm.py
│           └── train_temporal_trajectory.py
├── output/
│   └── feature_analysis/
│       └── feature_ranking.csv
├── src/
│   ├── benchmark/
│   │   ├── benchmark.py
│   │   ├── diag.py
│   │   ├── lightbench.py
│   │   ├── survival.py
│   │   └── output/
│   │       ├── benchmark/
│   │       ├── lgbm/
│   │       └── survival/
│   ├── preprocessing/
│   │   ├── build_features.py
│   │   └── segment_storm.py
│   └── test_direction/
│       ├── data.csv
│       ├── storm_direction_analysis.py
│       ├── test.html
│       └── output/
│           └── [fichiers HTML d'analyse de direction]
├── features_analysis.py
├── run_pipeline.ipynb
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
torch
lifelines (survival)
matplotlib, shap
```

