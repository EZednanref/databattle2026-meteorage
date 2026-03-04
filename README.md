![](./gif.gif)


Antoine : 

On va tenter une approche "probabilistic nowcasting" pour faire du forecasting de séries temporelles.

### Étape 1 — Borne supérieure (classification binaire)
"L'orage sera-t-il terminé avant un temps T ?"
  Modèles :

  - XGBoost / LightGBM — très utilisés sur données tabulaires météo, robustes, interprétables via SHAP
  - LSTM / GRU — si tu veux capturer la dynamique temporelle de la séquence d'éclairs (amplitude, distance, fréquence)
  - Survival Analysis (Cox, Weibull AFT) — particulièrement adapté : modélise le temps jusqu'à un événement (fin d'orage), gère nativement l'incertitude et les données censurées

### Étape 2 — Distribution temporelle (régression probabiliste)
"À quelle heure la probabilité que l'orage se termine est la plus haute ?"

  - Quantile Regression (XGBoost quantile loss) — donne directement des intervalles de confiance
  - Bayesian Neural Networks / MC Dropout — produisent une distribution de sortie
  - Conformalized Quantile Regression (CQR) — état de l'art pour les intervalles de prédiction calibrés, très utilisé récemment
  - NGBoost (Duan et al., 2019) — gradient boosting qui sort directement une gaussienne paramétrique


### Papiers de référence

- Probabilistic thunderstorm nowcasting using deep learning (Shi et al., 2017 — ConvLSTM)
- A machine learning approach to lightning prediction (Mostajabi et al., 2019, Nature npj Climate)
- Conformalized Quantile Regression (Romano et al., NeurIPS 2019)
- NGBoost: Natural Gradient Boosting (Duan et al., ICML 2020)
- Survival analysis for storm duration
