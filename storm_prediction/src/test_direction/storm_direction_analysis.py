"""
Analyse de la direction et vitesse de déplacement des orages.
Génère une visualisation HTML interactive pour chaque orage.

Usage:
    python storm_direction_analysis.py [--storm_id STORM_ID] [--min_duration MIN] [--output OUTPUT_DIR]
"""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CSV_PATH = "../../data/raw/data_with_storm_id.csv"
OUTPUT_DIR = Path("output")
GAP_MINUTES = 2          # Fenêtre temporelle pour barycentre (minutes)
META_GROUP_SIZE = 5      # Nombre de centroïdes par méta-groupe pour direction globale
MIN_DURATION_MINUTES = 1 # Durée minimale d'un orage pour l'analyser
AIRPORT_RADIUS_KM = 30   # Rayon de surveillance autour de l'aéroport
SUMMARY_MAX_ROWS = 100   # Limite de lignes dans output/index.html (évite un HTML trop lourd)

# RANSAC (trajectoire robuste)
RANSAC_MIN_POINTS = 4
RANSAC_MIN_SAMPLES_FRAC = 0.6
RANSAC_RESIDUAL_THRESHOLD_KM = 2.0
RANSAC_WINDOW = 20        # Nombre max de centroïdes récents utilisés pour RANSAC
                          # (les 20 derniers = ~40 min). Un orage long change de cap :
                          # seule la trajectoire récente prédit la sortie.

# Coordonnées des aéroports
AIRPORTS = {
    "Ajaccio":  {"lat": 41.9236, "lon": 8.8029},
    "Bastia":   {"lat": 42.5527, "lon": 9.4837},
    "Nantes":   {"lat": 47.1532, "lon": -1.6107},
    "Pise":     {"lat": 43.695,  "lon": 10.399},
    "Biarritz": {"lat": 43.4683, "lon": -1.524},
}

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en km entre deux points GPS."""
    R = 6371  # Rayon de la Terre en km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def linregress_simple(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Régression linéaire simple sans scipy.
    Returns: (slope, r_squared)
    """
    if len(x) < 2:
        return 0.0, 0.0
    
    n = len(x)
    mx, my = x.mean(), y.mean()
    ssxy = np.sum((x - mx) * (y - my))
    ssxx = np.sum((x - mx) ** 2)
    ssyy = np.sum((y - my) ** 2)
    
    slope = ssxy / ssxx if ssxx != 0 else 0
    r = ssxy / np.sqrt(ssxx * ssyy) if ssxx * ssyy != 0 else 0
    
    return slope, r ** 2


def analyze_trends(df_storm: pd.DataFrame) -> dict:
    """
    Analyse les tendances d'azimut et de distance par rapport à l'aéroport.
    
    Méthode hybride utilisant:
    - Tendance de distance: prédit si l'orage s'éloigne ou se rapproche
    - Tendance d'azimut: prédit comment l'orage tourne autour de l'aéroport
    - R²: mesure la fiabilité de la prédiction
    
    Returns:
        dict avec:
        - dist_slope: pente de la distance (km/min), >0 = s'éloigne
        - dist_r2: coefficient de détermination
        - azimuth_slope: pente de l'azimut (°/min), >0 = tourne sens horaire
        - azimuth_r2: coefficient de détermination
        - mean_azimuth: azimut moyen
        - mean_dist: distance moyenne
        - confidence: score de confiance global (0-1)
        - trend_label: 'S\'éloigne', 'Se rapproche', 'Stationnaire'
    """
    if len(df_storm) < 5 or 'azimuth' not in df_storm.columns or 'dist' not in df_storm.columns:
        return None
    
    df = df_storm.sort_values('date').copy()
    df['minutes'] = (df['date'] - df['date'].min()).dt.total_seconds() / 60
    
    # Régression sur la distance
    dist_slope, dist_r2 = linregress_simple(
        df['minutes'].values, 
        df['dist'].values
    )
    
    # Régression sur l'azimut (attention aux discontinuités 0/360)
    # On utilise l'azimut brut si pas de discontinuité majeure
    azimuths = df['azimuth'].values
    azimuth_range = azimuths.max() - azimuths.min()
    
    # Si l'écart est > 300°, l'orage traverse le 0/360, on ajuste
    if azimuth_range > 300:
        # Décaler les angles > 180 en négatif
        azimuths = np.where(azimuths > 180, azimuths - 360, azimuths)
    
    azimuth_slope, azimuth_r2 = linregress_simple(
        df['minutes'].values,
        azimuths
    )
    
    # Score de confiance combiné
    # Pondéré par la qualité des régressions
    confidence = (dist_r2 * 0.6 + azimuth_r2 * 0.4)  # Distance plus importante
    
    # Label de tendance
    if dist_slope > 0.05 and dist_r2 > 0.15:
        trend_label = "S'éloigne"
    elif dist_slope < -0.05 and dist_r2 > 0.15:
        trend_label = "Se rapproche"
    else:
        trend_label = "Stationnaire"
    
    return {
        'dist_slope': round(dist_slope, 4),          # km/min
        'dist_r2': round(dist_r2, 3),
        'azimuth_slope': round(azimuth_slope, 3),    # °/min
        'azimuth_r2': round(azimuth_r2, 3),
        'mean_azimuth': round(df['azimuth'].mean(), 1),
        'mean_dist': round(df['dist'].mean(), 2),
        'confidence': round(confidence, 3),
        'trend_label': trend_label,
        'dist_start': round(df['dist'].iloc[0], 2),
        'dist_end': round(df['dist'].iloc[-1], 2),
        'azimuth_start': round(df['azimuth'].iloc[0], 1),
        'azimuth_end': round(df['azimuth'].iloc[-1], 1),
    }


def get_cardinal_direction(angle_deg: float) -> str:
    """Convertit un angle en direction cardinale."""
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    idx = round(((angle_deg + 360) % 360) / 45) % 8
    return dirs[idx]


def circular_mean_deg(angles_deg: List[float], weights: Optional[List[float]] = None) -> Optional[float]:
  """Moyenne circulaire d'angles en degrés (0-360), robuste au passage 359→0."""
  if not angles_deg:
    return None

  if weights is None:
    weights = [1.0] * len(angles_deg)

  if len(weights) != len(angles_deg):
    raise ValueError("weights and angles_deg must have the same length")

  x = 0.0
  y = 0.0
  for angle, w in zip(angles_deg, weights):
    rad = math.radians(angle)
    x += w * math.cos(rad)
    y += w * math.sin(rad)

  if x == 0.0 and y == 0.0:
    return None

  mean_rad = math.atan2(y, x)
  mean_deg = (math.degrees(mean_rad) + 360) % 360
  return mean_deg


def project_latlon_to_xy_km(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Projection équirectangulaire locale vers (x,y) en km autour d'un point de référence."""
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * km_per_deg_lon
    y = (lat - ref_lat) * km_per_deg_lat
    return x, y


def unproject_xy_km_to_latlon(x_km: float, y_km: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Inverse de project_latlon_to_xy_km."""
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(ref_lat))
    lat = ref_lat + (y_km / km_per_deg_lat)
    lon = ref_lon + (x_km / km_per_deg_lon) if km_per_deg_lon != 0 else ref_lon
    return lat, lon


def solve_line_circle_exit_time(
    ax: float, bx: float, ay: float, by: float,
    t_now_min: float, radius_km: float
) -> Optional[float]:
    """Trouve le plus petit t >= t_now tel que (x(t)^2 + y(t)^2 = R^2)."""
    a = bx * bx + by * by
    if a <= 0:
        return None

    b = 2.0 * (ax * bx + ay * by)
    c = ax * ax + ay * ay - radius_km * radius_km
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= t_now_min]
    return min(candidates) if candidates else None


def fit_ransac_trajectory(
    centroids: list,
    airport_coords: dict,
    start_time: pd.Timestamp,
    radius_km: float,
) -> Optional[dict]:
    """Ajuste une trajectoire robuste (RANSAC) sur les barycentres récents (centroïdes 2 min).

    Seuls les RANSAC_WINDOW derniers centroïdes sont utilisés : la direction récente
    est ce qui prédit la sortie, pas la trajectoire depuis le début de l'orage.
    Un orage de 400 min a changé de cap plusieurs fois — utiliser tous les centroïdes
    noierait le signal récent et ferait chuter l'inlier_ratio artificiellement.
    """
    if not airport_coords or len(centroids) < RANSAC_MIN_POINTS:
        return None

    # Fenêtre glissante : ne garder que les RANSAC_WINDOW derniers centroïdes
    window = centroids[-RANSAC_WINDOW:]
    n_total = len(centroids)  # conservé pour info

    t = np.array([
        (pd.Timestamp(c['ts']) - start_time).total_seconds() / 60.0
        for c in window
    ], dtype=float)

    ref_lat, ref_lon = airport_coords['lat'], airport_coords['lon']
    xy = np.array([project_latlon_to_xy_km(c['lat'], c['lon'], ref_lat, ref_lon) for c in window], dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]

    if len(t) < 2 or np.allclose(t.max(), t.min()):
        return None

    t2d = t.reshape(-1, 1)  # conservé pour compatibilité r2 inlier scoring
    min_samples = max(2, int(math.ceil(RANSAC_MIN_SAMPLES_FRAC * len(t))))

    # Seuil adaptatif : 15% de la distance inter-barycentres médiane.
    # Les barycentres sont déjà lissés (agrégation d'éclairs sur 2 min) → bruit ~0.1-0.3 km.
    # Un changement de cap authentique dévie les barycentres de plusieurs × step,
    # donc 15% du step est discriminant sans être trop strict.
    dists_between = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    median_step_km = float(np.median(dists_between)) if len(dists_between) > 0 else 1.0
    adaptive_threshold = max(0.3, min(RANSAC_RESIDUAL_THRESHOLD_KM, median_step_km * 0.15))

    # --- RANSAC 2D par distance perpendiculaire ---
    # Contrairement à deux régressions scalaires séparées x(t)/y(t), ce RANSAC
    # utilise la distance perpendiculaire à la droite de trajectoire dans le plan (x,y).
    # Avantage : un orage qui change de cap produit des barycentres loin de la droite
    # dominante → véritables outliers détectés même si x(t) et y(t) restent quasi-linéaires
    # séparément.
    pts2d = np.column_stack([x, y])
    rng = np.random.default_rng(0)
    n_ransac_iter = max(100, 10 * len(t))
    best_inlier_mask = np.ones(len(t), dtype=bool)
    best_inlier_count = 0

    for _ in range(n_ransac_iter):
        idx = rng.choice(len(t), min_samples, replace=False)
        sample = pts2d[idx]
        centroid_s = sample.mean(axis=0)
        diffs_s = sample - centroid_s
        cov_s = diffs_s.T @ diffs_s
        try:
            _, vecs = np.linalg.eigh(cov_s)
        except np.linalg.LinAlgError:
            continue
        direction = vecs[:, -1]  # vecteur propre de la plus grande valeur propre

        # Distance perpendiculaire de tous les points à la droite
        diff_all = pts2d - centroid_s
        proj = diff_all @ direction
        perp = diff_all - np.outer(proj, direction)
        perp_dist = np.sqrt((perp ** 2).sum(axis=1))

        mask = perp_dist < adaptive_threshold
        count = int(mask.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = mask

    inlier_mask = best_inlier_mask
    inlier_ratio = float(inlier_mask.mean()) if len(inlier_mask) else 0.0

    # OLS sur les inliers pour obtenir ax, bx, ay, by (utilisés pour prédiction de sortie)
    t_fit = t[inlier_mask]
    x_fit = x[inlier_mask]
    y_fit = y[inlier_mask]

    if len(t_fit) >= 2 and not np.allclose(t_fit.max(), t_fit.min()):
        bx, ax = np.polyfit(t_fit, x_fit, 1)
        by, ay = np.polyfit(t_fit, y_fit, 1)
        x_pred = ax + bx * t_fit
        y_pred = ay + by * t_fit
        ssx = float(((x_fit - x_pred) ** 2).sum())
        ssy = float(((y_fit - y_pred) ** 2).sum())
        varx = float(((x_fit - x_fit.mean()) ** 2).sum())
        vary = float(((y_fit - y_fit.mean()) ** 2).sum())
        r2_x = 1.0 - (ssx / varx) if varx > 0 else 0.0
        r2_y = 1.0 - (ssy / vary) if vary > 0 else 0.0
    else:
        bx, ax = np.polyfit(t, x, 1)
        by, ay = np.polyfit(t, y, 1)
        inlier_mask = np.ones(len(t), dtype=bool)
        inlier_ratio = 1.0
        r2_x = r2_y = 0.0

    speed_kmh = float(math.sqrt(bx * bx + by * by) * 60.0)
    direction_angle = float(math.degrees(math.atan2(bx, by)))  # 0=N, 90=E
    direction = get_cardinal_direction(direction_angle)

    t_start = float(t.min())
    t_end = float(t.max())

    distance_km = float(math.sqrt(bx * bx + by * by) * (t_end - t_start))

    t_now = t_end
    x_now = ax + bx * t_now
    y_now = ay + by * t_now
    dist_now = float(math.sqrt(x_now * x_now + y_now * y_now))

    predicted_exit = None
    if dist_now < radius_km and speed_kmh > 1.0:
        t_exit = solve_line_circle_exit_time(ax, bx, ay, by, t_now_min=t_now, radius_km=radius_km)
        if t_exit is not None:
            x_exit = ax + bx * t_exit
            y_exit = ay + by * t_exit
            exit_lat, exit_lon = unproject_xy_km_to_latlon(x_exit, y_exit, ref_lat, ref_lon)
            exit_time = (start_time + pd.Timedelta(minutes=float(t_exit))).isoformat()
            predicted_exit = {
                'exit_lat': round(float(exit_lat), 5),
                'exit_lon': round(float(exit_lon), 5),
                'time_to_exit_min': round(float(t_exit - t_now), 1),
                'exit_distance_km': round(float(radius_km), 2),
                'predicted': True,
                'method': 'ransac',
                'exit_time': exit_time,
            }

    r2_mean = max(0.0, min(1.0, (r2_x + r2_y) / 2.0))

    # --- Validation azimutale ---
    # Calculer l'azimut de chaque barycentre inlier depuis l'aéroport (0=N, 90=E)
    # et mesurer à quel point il évolue de façon linéaire dans le temps.
    # Un orage cohérent a un azimut qui varie régulièrement → R²_azimut élevé.
    r2_azimut = 0.0
    t_in = t[inlier_mask]
    x_in = x[inlier_mask]
    y_in = y[inlier_mask]
    if len(t_in) >= 3:
        # Azimut brut en degrés (0=N, sens horaire)
        az_rad = np.arctan2(x_in, y_in)  # atan2(x,y) → 0=N
        az_deg = np.degrees(az_rad) % 360.0
        # Correction discontinuité 0°/360°: si l'écart max > 300°, décaler les >180°
        if az_deg.max() - az_deg.min() > 300.0:
            az_deg = np.where(az_deg > 180.0, az_deg - 360.0, az_deg)
        # Régression linéaire azimut ~ temps (OLS simple)
        t_in_c = t_in - t_in.mean()
        az_c = az_deg - az_deg.mean()
        ssxt = float(np.sum(t_in_c * az_c))
        sstt = float(np.sum(t_in_c ** 2))
        ssaz = float(np.sum(az_c ** 2))
        if sstt > 0 and ssaz > 0:
            r2_azimut = max(0.0, min(1.0, float((ssxt ** 2) / (sstt * ssaz))))

    # Confiance finale : linéarité cartésienne + cohérence azimutale
    # L'azimut sert de validateur : si la droite RANSAC est cohérente vue depuis
    # l'aéroport (azimut linéaire), la confiance est confirmée.
    confidence = max(0.0, min(1.0, inlier_ratio * (0.6 * r2_mean + 0.4 * r2_azimut)))
    if predicted_exit is not None:
        predicted_exit['confidence'] = round(float(confidence), 3)

    lat0, lon0 = unproject_xy_km_to_latlon(ax + bx * t_start, ay + by * t_start, ref_lat, ref_lon)
    lat1, lon1 = unproject_xy_km_to_latlon(ax + bx * t_end, ay + by * t_end, ref_lat, ref_lon)
    last_lat, last_lon = unproject_xy_km_to_latlon(x_now, y_now, ref_lat, ref_lon)

    return {
        'method': 'ransac',
        'n_points': int(len(t)),
        'inlier_ratio': round(float(inlier_ratio), 3),
        'r2_x': round(float(r2_x), 3),
        'r2_y': round(float(r2_y), 3),
        'r2_azimut': round(float(r2_azimut), 3),
        'confidence': round(float(confidence), 3),
        'adaptive_threshold_km': round(float(adaptive_threshold), 3),
        'n_window': len(window),
        'n_total_centroids': n_total,
        'speed_kmh': round(speed_kmh, 1),
        'direction': direction,
        'direction_angle': round(direction_angle, 2),
        'distance_km': round(distance_km, 2),
        'inlier_mask': [bool(v) for v in inlier_mask.tolist()],
        'line': {
            'start': {'lat': round(float(lat0), 5), 'lon': round(float(lon0), 5)},
            'end': {'lat': round(float(lat1), 5), 'lon': round(float(lon1), 5)},
            'last': {'lat': round(float(last_lat), 5), 'lon': round(float(last_lon), 5)},
        },
        'predicted_exit': predicted_exit,
    }


def robust_centroid(lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float) -> tuple:
    """
    Calcule un barycentre robuste en rejetant les éclairs outliers (RANSAC spatial).

    Principe :
    - Projeter les éclairs en (x, y) km autour du centre provisoire
    - Calculer la distance médiane de chaque éclair à la médiane spatiale (médiane de x, médiane de y)
    - Seuil = 2× la MAD (Median Absolute Deviation) des distances — robuste aux groupes denses
    - Les éclairs au-delà du seuil sont des outliers (cellule isolée, éclair errant)
    - Le barycentre final est la moyenne des inliers uniquement

    Returns:
        (lat_robust, lon_robust, inlier_fraction, n_inliers)
    """
    n = len(lats)
    if n <= 2:
        return float(lats.mean()), float(lons.mean()), 1.0, n

    xs = np.array([project_latlon_to_xy_km(lat, lon, ref_lat, ref_lon)[0] for lat, lon in zip(lats, lons)])
    ys = np.array([project_latlon_to_xy_km(lat, lon, ref_lat, ref_lon)[1] for lat, lon in zip(lats, lons)])

    # Médiane spatiale (L1 center)
    med_x = float(np.median(xs))
    med_y = float(np.median(ys))

    # Distance de chaque éclair au centre médian
    dists = np.sqrt((xs - med_x) ** 2 + (ys - med_y) ** 2)

    # MAD des distances (robuste) → seuil = 2× MAD (ou min 0.5 km)
    mad = float(np.median(np.abs(dists - np.median(dists))))
    threshold = max(0.5, 2.0 * mad)

    inlier_mask = dists <= threshold
    n_inliers = int(inlier_mask.sum())

    # Fallback : si moins de 2 inliers, garder la médiane brute
    if n_inliers < 2:
        return float(lats.mean()), float(lons.mean()), 1.0, n

    lat_robust = float(lats[inlier_mask].mean())
    lon_robust = float(lons[inlier_mask].mean())
    inlier_frac = float(n_inliers / n)

    return lat_robust, lon_robust, inlier_frac, n_inliers


def predict_exit_point(
    current_lat: float, current_lon: float,
    direction_deg: float, speed_kmh: float,
    airport_lat: float, airport_lon: float,
    radius_km: float
) -> dict:
    """
    Prédit quand et où l'orage sortira du cercle de surveillance.
    
    Utilise une projection linéaire basée sur:
    - Position actuelle du centroïde
    - Direction de déplacement (angle en degrés, 0=Nord, 90=Est)
    - Vitesse de déplacement (km/h)
    
    Returns:
        dict avec exit_lat, exit_lon, time_to_exit_min, exit_distance_km
        ou None si l'orage ne sortira pas (se dirige vers le centre ou trop lent)
    """
    if speed_kmh <= 0:
        return None
    
    # Distance actuelle à l'aéroport
    current_dist = haversine(current_lat, current_lon, airport_lat, airport_lon)
    
    # Si déjà hors du cercle, pas de prédiction
    if current_dist >= radius_km:
        return None
    
    # Convertir direction en radians (0=Nord, sens horaire)
    # En math standard: 0=Est, sens anti-horaire
    # Conversion: math_angle = 90 - direction_deg
    direction_rad = math.radians(90 - direction_deg)
    
    # Vecteur de direction unitaire (en coordonnées approximatives)
    # 1 degré de latitude ≈ 111 km
    # 1 degré de longitude ≈ 111 * cos(lat) km
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(current_lat))
    
    # Vecteur de déplacement en degrés par heure
    dx_deg_per_h = (speed_kmh * math.cos(direction_rad)) / km_per_deg_lon  # longitude
    dy_deg_per_h = (speed_kmh * math.sin(direction_rad)) / km_per_deg_lat  # latitude
    
    # Recherche par itération du point de sortie
    # On avance pas à pas jusqu'à sortir du cercle
    dt_hours = 1 / 60  # Pas de 1 minute
    max_hours = 24  # Maximum 24h de prédiction
    
    t = 0
    lat, lon = current_lat, current_lon
    
    while t < max_hours:
        t += dt_hours
        lat += dy_deg_per_h * dt_hours
        lon += dx_deg_per_h * dt_hours
        
        dist = haversine(lat, lon, airport_lat, airport_lon)
        
        if dist >= radius_km:
            # Trouvé le point de sortie
            return {
                'exit_lat': round(lat, 5),
                'exit_lon': round(lon, 5),
                'time_to_exit_min': round(t * 60, 1),
                'exit_distance_km': round(dist, 2),
                'predicted': True
            }
    
    # Pas de sortie prévue dans les 24h (orage trop lent ou se dirige vers le centre)
    return None


# ============================================================================
# Analyse d'un orage
# ============================================================================

def analyze_storm(df_storm: pd.DataFrame, storm_id: str) -> dict:
    """
    Analyse un orage et retourne les métriques de direction/vitesse.
    
    Utilise une approche hybride combinant:
    1. Méthode centroïdes: direction géographique réelle (N/S/E/O)
    2. Tendance distance: prédit si l'orage s'éloigne/se rapproche de l'aéroport
    3. Tendance azimut: prédit comment l'orage tourne autour de l'aéroport
    
    Returns:
        dict avec:
        - storm_id, airport, duration_min
        - n_strikes: nombre total d'éclairs
        - centroids: liste des centroïdes par fenêtre temporelle
        - meta_centroids: centroïdes agrégés par META_GROUP_SIZE
        - global_speed_kmh: vitesse moyenne globale
        - global_direction: direction cardinale (N, NE, etc.)
        - global_distance_km: distance totale parcourue
        - trends: analyse des tendances (azimut, distance, confiance)
    """
    df = df_storm.sort_values('date').copy()
    
    if len(df) == 0:
        return None
    
    airport = df['airport'].iloc[0] if 'airport' in df.columns else 'Unknown'
    start_time = df['date'].min()
    end_time = df['date'].max()
    duration_min = (end_time - start_time).total_seconds() / 60
    
    # === Grouper par fenêtres temporelles ===
    df['time_group'] = (df['date'] - start_time).dt.total_seconds() // (GAP_MINUTES * 60)
    
    # Récupérer les coordonnées de l'aéroport
    airport_coords = AIRPORTS.get(airport, None)
    
    centroids = []
    for group_id, group in df.groupby('time_group'):
        lats = group['lat'].values
        lons = group['lon'].values

        if airport_coords:
            lat_c, lon_c, inlier_frac, n_inliers = robust_centroid(
                lats, lons, airport_coords['lat'], airport_coords['lon']
            )
        else:
            lat_c, lon_c = float(lats.mean()), float(lons.mean())
            inlier_frac, n_inliers = 1.0, len(lats)

        centroid = {
            'lat': lat_c,
            'lon': lon_c,
            'ts': group['date'].median().isoformat(),
            'n': len(group),
            'n_inliers': n_inliers,
            'inlier_frac': round(inlier_frac, 3),
            'amp_mean': group['amplitude'].mean() if 'amplitude' in group.columns else 0,
            'azimuth_mean': round(group['azimuth'].mean(), 1) if 'azimuth' in group.columns else None,
        }
        
        # Calculer la distance par rapport à l'aéroport
        if airport_coords:
            centroid['dist_to_airport'] = haversine(
                centroid['lat'], centroid['lon'],
                airport_coords['lat'], airport_coords['lon']
            )
            centroid['inside_radius'] = centroid['dist_to_airport'] <= AIRPORT_RADIUS_KM
        
        centroids.append(centroid)
    
    # === Identifier le moment de sortie de la zone des 30km ===
    exit_info = None
    if airport_coords and len(centroids) >= 2:
        # Chercher le premier centroïde qui sort de la zone
        was_inside = centroids[0].get('inside_radius', True)
        for i, c in enumerate(centroids):
            is_inside = c.get('inside_radius', True)
            if was_inside and not is_inside:
                # L'orage vient de sortir !
                exit_info = {
                    'exit_centroid_idx': i,
                    'exit_time': c['ts'],
                    'exit_dist_km': round(c['dist_to_airport'], 2),
                    'time_to_exit_min': round((pd.Timestamp(c['ts']) - start_time).total_seconds() / 60, 1),
                }
                break
            was_inside = is_inside
    
    if len(centroids) < 2:
        return {
            'storm_id': storm_id,
            'airport': airport,
            'airport_coords': airport_coords,
            'duration_min': duration_min,
            'n_strikes': len(df),
            'centroids': centroids,
            'meta_centroids': centroids,
            'global_speed_kmh': 0,
            'global_direction': '—',
            'global_distance_km': 0,
            'exit_info': exit_info,
        }
    
    # === Calculer distances et vitesses entre centroïdes ===
    for i in range(1, len(centroids)):
        c0, c1 = centroids[i-1], centroids[i]
        d = haversine(c0['lat'], c0['lon'], c1['lat'], c1['lon'])
        centroids[i]['dist_from_prev'] = d
        
        dt_hours = (pd.Timestamp(c1['ts']) - pd.Timestamp(c0['ts'])).total_seconds() / 3600
        centroids[i]['speed_kmh'] = d / dt_hours if dt_hours > 0 else 0
    
    centroids[0]['dist_from_prev'] = 0
    centroids[0]['speed_kmh'] = 0
    
    # === Méta-centroïdes (agrégés par META_GROUP_SIZE) ===
    meta_centroids = []
    for i in range(0, len(centroids), META_GROUP_SIZE):
        group = centroids[i:i + META_GROUP_SIZE]
        total_n = sum(c['n'] for c in group)

        azimuth_values = [c['azimuth_mean'] for c in group if c.get('azimuth_mean') is not None]
        azimuth_weights = [c['n'] for c in group if c.get('azimuth_mean') is not None]
        meta_azimuth = circular_mean_deg(azimuth_values, azimuth_weights) if azimuth_values else None
        
        meta = {
            'lat': sum(c['lat'] * c['n'] for c in group) / total_n,
            'lon': sum(c['lon'] * c['n'] for c in group) / total_n,
            'ts': group[len(group)//2]['ts'],  # Timestamp du milieu
            'n': total_n,
            'group_range': [i + 1, min(i + META_GROUP_SIZE, len(centroids))],
            'azimuth_mean': round(meta_azimuth, 1) if meta_azimuth is not None else None,
        }
        meta_centroids.append(meta)
    
    # === Calculer métriques globales (baseline via méta-centroïdes) ===
    global_dist = 0
    global_time_hours = 0
    
    for i in range(1, len(meta_centroids)):
        m0, m1 = meta_centroids[i-1], meta_centroids[i]
        d = haversine(m0['lat'], m0['lon'], m1['lat'], m1['lon'])
        global_dist += d
        
        dt = (pd.Timestamp(m1['ts']) - pd.Timestamp(m0['ts'])).total_seconds() / 3600
        global_time_hours += dt
        
        meta_centroids[i]['dist_from_prev'] = d
        meta_centroids[i]['speed_kmh'] = d / dt if dt > 0 else 0
    
    meta_centroids[0]['dist_from_prev'] = 0
    meta_centroids[0]['speed_kmh'] = 0
    
    global_speed = global_dist / global_time_hours if global_time_hours > 0 else 0
    
    # Direction globale (du premier au dernier méta-centroïde)
    direction_angle = None
    if len(meta_centroids) >= 2:
        first, last = meta_centroids[0], meta_centroids[-1]
        # Angle en degrés (0=Nord, 90=Est, convention géographique)
        direction_angle = math.degrees(math.atan2(last['lon'] - first['lon'], last['lat'] - first['lat']))
        direction = get_cardinal_direction(direction_angle)
    else:
        direction = '—'
    
    # === Analyse des tendances (azimut et distance) ===
    trends = analyze_trends(df_storm)

    # === Trajectoire robuste via RANSAC (barycentres 2 minutes) ===
    ransac = None
    if airport_coords and len(centroids) >= RANSAC_MIN_POINTS:
      ransac = fit_ransac_trajectory(
        centroids=centroids,
        airport_coords=airport_coords,
        start_time=pd.Timestamp(start_time),
        radius_km=AIRPORT_RADIUS_KM,
      )

      # Si RANSAC est valide, on l'utilise comme métriques principales
      if ransac and ransac.get('speed_kmh', 0) > 0:
        global_speed = float(ransac['speed_kmh'])
        direction = ransac.get('direction', direction)
        direction_angle = float(ransac.get('direction_angle', direction_angle))
        global_dist = float(ransac.get('distance_km', global_dist))

    # === Prédiction de sortie de zone ===
    predicted_exit = None

    # 1) Priorité à RANSAC
    if ransac and ransac.get('predicted_exit'):
      predicted_exit = ransac['predicted_exit']

    # 2) Fallback (ancienne méthode) si pas de RANSAC
    if predicted_exit is None and airport_coords and len(meta_centroids) >= 2 and global_speed > 0 and direction_angle is not None:
      last_meta = meta_centroids[-1]
      last_dist = haversine(
        last_meta['lat'], last_meta['lon'],
        airport_coords['lat'], airport_coords['lon']
      )

      if last_dist < AIRPORT_RADIUS_KM:
        effective_speed = global_speed
        if trends and trends['confidence'] > 0.3:
          dist_speed_kmh = abs(trends['dist_slope']) * 60
          if dist_speed_kmh > 5:
            effective_speed = dist_speed_kmh

        predicted_exit = predict_exit_point(
          current_lat=last_meta['lat'],
          current_lon=last_meta['lon'],
          direction_deg=direction_angle,
          speed_kmh=effective_speed,
          airport_lat=airport_coords['lat'],
          airport_lon=airport_coords['lon'],
          radius_km=AIRPORT_RADIUS_KM,
        )

        if predicted_exit and trends:
          predicted_exit['confidence'] = trends['confidence']
          predicted_exit['method'] = 'hybrid' if trends['confidence'] > 0.3 else 'centroid'

        if predicted_exit:
          base_ts = pd.Timestamp(last_meta['ts'])
          predicted_ts = base_ts + pd.Timedelta(minutes=float(predicted_exit['time_to_exit_min']))
          predicted_exit['exit_time'] = predicted_ts.isoformat()
    
    return {
        'storm_id': storm_id,
        'airport': airport,
        'airport_coords': airport_coords,
        'duration_min': round(duration_min, 1),
        'n_strikes': len(df),
        'centroids': centroids,
        'meta_centroids': meta_centroids,
        'global_speed_kmh': round(float(global_speed), 1),
        'global_direction': direction,
        'direction_angle': direction_angle,
        'global_distance_km': round(float(global_dist), 2),
        'exit_info': exit_info,
        'predicted_exit': predicted_exit,
        'radius_km': AIRPORT_RADIUS_KM,
        'trends': trends,  # Nouvelle analyse hybride
        'ransac': ransac,
    }


# ============================================================================
# Génération HTML
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Storm Direction - {storm_id}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root {{
    --bg: #0a0c0f; --panel: #111418; --border: #1e2530;
    --accent: #00e5ff; --accent2: #ff4d6d; --accent3: #b2ff59;
    --text: #c8d6e5; --muted: #4a5568;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh; display: flex; flex-direction: column;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    padding: 12px 24px; background: var(--panel);
  }}
  header h1 {{ font-size: 18px; color: var(--accent); }}
  .stat {{ font-size: 12px; color: var(--muted); }}
  .stat span {{ color: var(--accent); font-weight: 700; }}
  .stat.highlight span {{ color: var(--accent3); }}
  #map {{ flex: 1; background: var(--bg); }}
  .leaflet-container {{ background: #0d1117; }}
  .custom-tooltip {{
    background: var(--panel); border: 1px solid var(--accent);
    padding: 8px 12px; font-size: 11px; line-height: 1.6;
  }}
  .custom-tooltip b {{ color: var(--accent); }}
  .info-box {{
    position: absolute; bottom: 20px; left: 20px; z-index: 1000;
    background: rgba(17,20,24,0.95); border: 1px solid var(--accent);
    padding: 16px 20px; font-size: 12px; line-height: 1.8;
  }}
  .info-box h3 {{ color: var(--accent); margin-bottom: 8px; font-size: 14px; }}
  .info-box .big {{ font-size: 24px; font-weight: 700; color: var(--accent3); }}
  .exit-box {{
    position: absolute; bottom: 20px; right: 20px; z-index: 1000;
    background: rgba(17,20,24,0.95); border: 1px solid var(--accent2);
    padding: 16px 20px; font-size: 12px; line-height: 1.8; max-width: 280px;
  }}
  .exit-box h3 {{ color: var(--accent2); margin-bottom: 8px; font-size: 14px; }}
  .exit-box .big {{ font-size: 20px; font-weight: 700; color: var(--accent2); }}
  .stat.exit span {{ color: var(--accent2); }}
  .stat.prediction span {{ color: #ffea00; }}
  .stat.trend span {{ color: #b388ff; }}
  .prediction-box {{
    position: absolute; top: 80px; right: 20px; z-index: 1000;
    background: rgba(17,20,24,0.95); border: 2px solid #ffea00;
    padding: 16px 20px; font-size: 12px; line-height: 1.8; max-width: 300px;
  }}
  .prediction-box h3 {{ color: #ffea00; margin-bottom: 8px; font-size: 14px; }}
  .prediction-box .big {{ font-size: 22px; font-weight: 700; color: #ffea00; }}
  .trends-box {{
    position: absolute; top: 80px; left: 20px; z-index: 1000;
    background: rgba(17,20,24,0.95); border: 1px solid #b388ff;
    padding: 16px 20px; font-size: 12px; line-height: 1.8; max-width: 320px;
  }}
  .trends-box h3 {{ color: #b388ff; margin-bottom: 8px; font-size: 14px; }}
  .trends-box .trend-value {{ font-weight: 700; }}
  .trends-box .trend-value.positive {{ color: #ff4d6d; }}
  .trends-box .trend-value.negative {{ color: #69f0ae; }}
  .trends-box .r2 {{ color: var(--muted); font-size: 10px; }}
  .confidence-bar {{ 
    height: 6px; background: #1e2530; border-radius: 3px; margin-top: 8px;
  }}
  .confidence-bar .fill {{ 
    height: 100%; border-radius: 3px; transition: width 0.3s;
  }}
</style>
</head>
<body>

<header>
  <h1>{storm_id}</h1>
  <div class="stat">Aéroport : <span>{airport}</span></div>
  <div class="stat">Durée : <span>{duration_min} min</span></div>
  <div class="stat">Éclairs : <span>{n_strikes}</span></div>
  <div class="stat">Barycentres (2min) : <span>{n_centroids}</span></div>
  <div class="stat">Méta-points : <span>{n_meta}</span></div>
  <div class="stat highlight">Distance : <span>{global_distance_km} km</span></div>
  <div class="stat highlight">Direction : <span>{global_speed_kmh} km/h → {global_direction}</span></div>
  <div class="stat">RANSAC : <span>{ransac_status}</span></div>
  <div class="stat prediction">Prédiction : <span>{prediction_status}</span></div>
</header>

<div id="map"></div>

<div class="trends-box" id="trends-box" style="display:{trends_box_display};">
  <h3>Analyse des tendances</h3>
  <div style="margin-bottom:10px;">
    <strong>Distance</strong> {dist_start} → {dist_end} km<br>
    <span class="trend-value {dist_trend_class}">{dist_slope_display}</span>
    <span class="r2">(R² = {dist_r2})</span>
  </div>
  <div style="margin-bottom:10px;">
    <strong>Azimut</strong> {azimuth_start}° → {azimuth_end}°<br>
    <span class="trend-value">{azimuth_slope_display}</span>
    <span class="r2">(R² = {azimuth_r2})</span>
  </div>
</div>

<div class="info-box">
  <h3>Direction globale</h3>
  <div class="big">{global_speed_kmh} km/h → {global_direction}</div>
  <div style="margin-top:8px;color:var(--muted);">
    Distance totale : {global_distance_km} km<br>
    {direction_basis}
  </div>
</div>

<div class="exit-box" id="exit-box" style="display:{exit_box_display};">
  <h3>Sortie zone {radius_km}km</h3>
  <div class="big">{exit_time_display}</div>
  <div style="margin-top:8px;color:var(--muted);">
    Après {exit_time_min} min<br>
    Distance aéroport : {exit_dist_km} km<br>
    Centroïde #{exit_centroid_idx}
  </div>
</div>

<div class="prediction-box" id="prediction-box" style="display:{prediction_box_display};">
  <h3>Prédiction de sortie</h3>
  <div class="big">Dans ~{predicted_time_min} min</div>
  <div style="margin-top:8px;color:var(--muted);">
    Position estimée :<br>
    {predicted_lat}, {predicted_lon}<br>
    Basé sur vitesse {global_speed_kmh} km/h → {global_direction}
  </div>
</div>

<script>
const DATA = {data_json};

const PALETTE = ['#00e5ff','#69f0ae','#ffea00','#ff9100','#ff4081'];
const META_COLORS = ['#00e5ff', '#69f0ae', '#ffea00', '#ff9100', '#ff4081'];

const map = L.map('map', {{ zoomControl: true }}).setView([DATA.centroids[0].lat, DATA.centroids[0].lon], 10);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© OpenStreetMap © CARTO', subdomains: 'abcd', maxZoom: 19
}}).addTo(map);

// Barycentres (petits points) - couleur change tous les 5
DATA.centroids.forEach((c, i) => {{
  const groupIndex = Math.floor(i / 5);  // Groupe de 5 centroïdes
  const color = PALETTE[groupIndex % PALETTE.length];
  // inlier_frac = fraction d'éclairs conservés lors du RANSAC spatial (filtre outliers éclairs)
  const inlierFrac = (c.inlier_frac !== undefined) ? c.inlier_frac : 1.0;
  const nInliers = (c.n_inliers !== undefined) ? c.n_inliers : c.n;
  const nOutliers = c.n - nInliers;
  const outline = inlierFrac >= 0.8 ? '#69f0ae' : (inlierFrac >= 0.5 ? '#ffea00' : '#ff4d6d');
  const inlierBadge = '';
  L.circleMarker([c.lat, c.lon], {{
    radius: 5, fillColor: color, color: outline, weight: 2, fillOpacity: 0.7
  }}).bindTooltip(`<div class="custom-tooltip">
    <b style="color:${{color}}">Barycentre ${{i+1}}</b> (groupe ${{groupIndex+1}}) ${{inlierBadge}}<br>
    ${{c.ts}}<br>
    Éclairs: <b>${{nInliers}}</b>/${{c.n}} conservés (${{Math.round(inlierFrac*100)}}%)
    ${{nOutliers > 0 ? ' · <span style="color:#ff4d6d">' + nOutliers + ' outlier(s) rejetés</span>' : ''}}<br>
    ${{c.azimuth_mean !== null ? 'Azimut: ' + c.azimuth_mean.toFixed(1) + '°<br>' : ''}}
    ${{c.dist_from_prev ? 'Dist: ' + c.dist_from_prev.toFixed(2) + ' km' : ''}}
    ${{c.speed_kmh ? ' · ' + c.speed_kmh.toFixed(1) + ' km/h' : ''}}
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);
}});

// Ligne entre centroïdes (fine, grise)
if (DATA.centroids.length > 1) {{
  const line = DATA.centroids.map(c => [c.lat, c.lon]);
  L.polyline(line, {{ color: '#4a5568', weight: 1.5, opacity: 0.5 }}).addTo(map);
}}

// Méta-centroïdes (gros points)
DATA.meta_centroids.forEach((mc, i) => {{
  const color = META_COLORS[i % META_COLORS.length];
  
  L.circleMarker([mc.lat, mc.lon], {{
    radius: 14, fillColor: color, color: '#ffffff', weight: 3, fillOpacity: 0.95
  }}).bindTooltip(`<div class="custom-tooltip">
    <b style="color:${{color}}">◆ Direction ${{i+1}}/${{DATA.meta_centroids.length}}</b><br>
    ${{mc.ts}}<br>
    Groupes ${{mc.group_range[0]}}→${{mc.group_range[1]}}<br>
    Éclairs: ${{mc.n}}<br>
    ${{mc.azimuth_mean !== null ? 'Azimut: ' + mc.azimuth_mean.toFixed(1) + '°<br>' : ''}}
    ${{mc.dist_from_prev ? 'Dist: ' + mc.dist_from_prev.toFixed(2) + ' km · ' + mc.speed_kmh.toFixed(1) + ' km/h' : ''}}
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);

  // Numéro
  L.marker([mc.lat, mc.lon], {{
    icon: L.divIcon({{
      html: `<div style="font-size:11px;font-weight:800;color:#0a0c0f;text-align:center;line-height:14px;">${{i+1}}</div>`,
      className: '', iconSize: [14,14], iconAnchor: [7,7]
    }})
  }}).addTo(map);
}});

// Ligne de direction globale (épaisse, blanche)
if (DATA.meta_centroids.length > 1) {{
  const dirLine = DATA.meta_centroids.map(mc => [mc.lat, mc.lon]);
  L.polyline(dirLine, {{ color: '#ffffff', weight: 4, opacity: 0.9 }}).addTo(map);

  // Flèche de direction
  const last = DATA.meta_centroids[DATA.meta_centroids.length - 1];
  const prev = DATA.meta_centroids[DATA.meta_centroids.length - 2];
  const angle = Math.atan2(last.lon - prev.lon, last.lat - prev.lat) * 180 / Math.PI;
  L.marker([last.lat, last.lon], {{
    icon: L.divIcon({{
      html: `<div style="transform:rotate(${{angle-90}}deg);font-size:28px;color:#ff4081;">➤</div>`,
      className: '', iconSize: [28,28], iconAnchor: [14,14]
    }})
  }}).addTo(map);
  
  // Labels distance sur segments
  for (let i = 1; i < DATA.meta_centroids.length; i++) {{
    const m0 = DATA.meta_centroids[i-1];
    const m1 = DATA.meta_centroids[i];
    const midLat = (m0.lat + m1.lat) / 2;
    const midLon = (m0.lon + m1.lon) / 2;
    const color = META_COLORS[i % META_COLORS.length];
    L.marker([midLat, midLon], {{
      icon: L.divIcon({{
        html: `<div style="background:rgba(10,12,15,0.9);border:1px solid ${{color}};padding:2px 6px;font-size:9px;color:${{color}};white-space:nowrap;">
          ${{m1.dist_from_prev.toFixed(1)}} km · ${{m1.speed_kmh.toFixed(0)}} km/h
        </div>`,
        className: '', iconSize: null, iconAnchor: [40, 10]
      }})
    }}).addTo(map);
  }}
}}

// === Trajectoire RANSAC (robuste) ===
if (DATA.ransac && DATA.ransac.line) {{
  const s = DATA.ransac.line.start;
  const e = DATA.ransac.line.end;
  L.polyline([[s.lat, s.lon], [e.lat, e.lon]], {{
    color: '#69f0ae', weight: 4, opacity: 0.95, dashArray: '10 6'
  }}).bindTooltip(`<div class="custom-tooltip">
    <b style="color:#69f0ae">RANSAC</b><br>
    Inliers: ${{Math.round(DATA.ransac.inlier_ratio * 100)}}%<br>
    Confiance: ${{Math.round(DATA.ransac.confidence * 100)}}%<br>
    Vitesse: ${{DATA.ransac.speed_kmh}} km/h → ${{DATA.ransac.direction}}
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);
}}

// === Cercle de 30km autour de l'aéroport ===
if (DATA.airport_coords) {{
  // Cercle de la zone de surveillance
  L.circle([DATA.airport_coords.lat, DATA.airport_coords.lon], {{
    radius: DATA.radius_km * 1000,
    color: '#ff4d6d',
    fillColor: '#ff4d6d',
    fillOpacity: 0.05,
    weight: 2,
    dashArray: '8 4'
  }}).bindTooltip(`<div class="custom-tooltip">
    <b style="color:#ff4d6d">Zone de surveillance</b><br>
    Rayon : ${{DATA.radius_km}} km
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);

  // Marqueur de l'aéroport
  L.marker([DATA.airport_coords.lat, DATA.airport_coords.lon], {{
    icon: L.divIcon({{
      html: `<div style="font-size:20px;">✈️</div>`,
      className: '', iconSize: [20,20], iconAnchor: [10,10]
    }})
  }}).bindTooltip(`<div class="custom-tooltip">
    <b>Aéroport ${{DATA.airport}}</b><br>
    ${{DATA.airport_coords.lat.toFixed(4)}}, ${{DATA.airport_coords.lon.toFixed(4)}}
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);
}}

// === Marqueur du point de sortie de zone ===
if (DATA.exit_info) {{
  const exitCentroid = DATA.centroids[DATA.exit_info.exit_centroid_idx];
  
  // Cercle rouge pour marquer la sortie
  L.circleMarker([exitCentroid.lat, exitCentroid.lon], {{
    radius: 18,
    fillColor: 'transparent',
    color: '#ff4d6d',
    weight: 4,
    fillOpacity: 0
  }}).addTo(map);
  
  // Label "SORTIE"
  L.marker([exitCentroid.lat, exitCentroid.lon], {{
    icon: L.divIcon({{
      html: `<div style="background:#ff4d6d;color:white;padding:2px 8px;font-size:10px;font-weight:bold;border-radius:2px;white-space:nowrap;transform:translateY(-25px);">SORTIE</div>`,
      className: '', iconSize: null, iconAnchor: [25, 0]
    }})
  }}).addTo(map);
}}

// === Point de sortie PRÉDIT ===
if (DATA.predicted_exit) {{
  const pred = DATA.predicted_exit;
  
  // Ligne de projection (du dernier méta-centroïde au point prédit)
  const fromPt = (DATA.ransac && DATA.ransac.line && DATA.ransac.line.last)
    ? DATA.ransac.line.last
    : DATA.meta_centroids[DATA.meta_centroids.length - 1];

  L.polyline([[fromPt.lat, fromPt.lon], [pred.exit_lat, pred.exit_lon]], {{
    color: '#ffea00',
    weight: 3,
    opacity: 0.8,
    dashArray: '6 4'
  }}).addTo(map);
  
  // Cercle jaune au point de sortie prédit
  L.circleMarker([pred.exit_lat, pred.exit_lon], {{
    radius: 12,
    fillColor: '#ffea00',
    color: '#ffffff',
    weight: 3,
    fillOpacity: 0.9
  }}).bindTooltip(`<div class="custom-tooltip">
    <b style="color:#ffea00">SORTIE PRÉDITE</b><br>
    Dans ~${{pred.time_to_exit_min}} min<br>
    ${{pred.exit_lat}}, ${{pred.exit_lon}}<br>
    Distance : ${{pred.exit_distance_km}} km de l'aéroport
  </div>`, {{ className: '', opacity: 1 }}).addTo(map);
  
  // Label "PRÉDIT"
  L.marker([pred.exit_lat, pred.exit_lon], {{
    icon: L.divIcon({{
      html: `<div style="background:#ffea00;color:#0a0c0f;padding:3px 10px;font-size:10px;font-weight:bold;border-radius:2px;white-space:nowrap;transform:translateY(-28px);">~${{pred.time_to_exit_min}} min</div>`,
      className: '', iconSize: null, iconAnchor: [35, 0]
    }})
  }}).addTo(map);
}}

// Colorer les centroïdes selon inside/outside
DATA.centroids.forEach((c, i) => {{
  if (c.dist_to_airport !== undefined) {{
    const isInside = c.inside_radius;
    const borderColor = isInside ? '#00e5ff' : '#ff4d6d';
    L.circleMarker([c.lat, c.lon], {{
      radius: 3,
      fillColor: borderColor,
      color: borderColor,
      weight: 1,
      fillOpacity: 0.3
    }}).addTo(map);
  }}
}});

// Ajuster la vue pour inclure l'aéroport
let bounds = L.latLngBounds(DATA.centroids.map(c => [c.lat, c.lon]));
if (DATA.airport_coords) {{
  bounds.extend([DATA.airport_coords.lat, DATA.airport_coords.lon]);
}}
map.fitBounds(bounds.pad(0.15));
</script>
</body>
</html>
'''


def generate_html(analysis: dict, output_path: Path) -> None:
    """Génère un fichier HTML pour visualiser un orage."""
    
    # Préparer les infos de sortie de zone
    exit_info = analysis.get('exit_info')
    if exit_info:
        exit_time_display = exit_info['exit_time'][:19].replace('T', ' ')
        exit_time_short = exit_info['exit_time'][11:16]  # HH:MM
        exit_time_min = exit_info['time_to_exit_min']
        exit_status = f"{exit_time_short} ({exit_time_min} min)"
        exit_box_display = "block"
        exit_dist_km = exit_info['exit_dist_km']
        exit_centroid_idx = exit_info['exit_centroid_idx'] + 1
    else:
        exit_status = "Reste dans la zone"
        exit_box_display = "none"
        exit_time_display = "—"
        exit_time_min = "—"
        exit_dist_km = "—"
        exit_centroid_idx = "—"
    
    # Préparer les infos de prédiction
    predicted_exit = analysis.get('predicted_exit')
    prediction_status = "—"
    prediction_box_display = "none"
    predicted_time_min = "—"
    predicted_lat = "—"
    predicted_lon = "—"

    if predicted_exit:
        prediction_box_display = "block"
        predicted_time_min = predicted_exit.get('time_to_exit_min', "—")
        predicted_lat = predicted_exit.get('exit_lat', "—")
        predicted_lon = predicted_exit.get('exit_lon', "—")

        if predicted_exit.get('exit_time'):
            pred_time_short = predicted_exit['exit_time'][11:16]  # HH:MM
            prediction_status = f"{pred_time_short} (~{predicted_exit['time_to_exit_min']} min)"
        else:
            prediction_status = f"~{predicted_exit['time_to_exit_min']} min"
    
    # Préparer les infos de tendances (méthode hybride)
    trends = analysis.get('trends')
    if trends:
        trends_box_display = "block"
        trend_label = trends['trend_label']
        dist_slope = trends['dist_slope']
        dist_r2 = trends['dist_r2']
        azimuth_slope = trends['azimuth_slope']
        azimuth_r2 = trends['azimuth_r2']
        confidence = trends['confidence']
        confidence_pct = int(confidence * 100)
        dist_start = trends['dist_start']
        dist_end = trends['dist_end']
        azimuth_start = trends['azimuth_start']
        azimuth_end = trends['azimuth_end']
        
        # Formater l'affichage de la pente distance
        if dist_slope > 0:
            dist_slope_display = f"+{dist_slope*60:.1f} km/h (s'éloigne)"
            dist_trend_class = "positive"
        elif dist_slope < 0:
            dist_slope_display = f"{dist_slope*60:.1f} km/h (se rapproche)"
            dist_trend_class = "negative"
        else:
            dist_slope_display = "0 km/h (stationnaire)"
            dist_trend_class = ""
        
        # Formater l'affichage de la pente azimut
        if azimuth_slope > 0:
            azimuth_slope_display = f"+{azimuth_slope:.2f}°/min (sens horaire)"
        elif azimuth_slope < 0:
            azimuth_slope_display = f"{azimuth_slope:.2f}°/min (sens anti-horaire)"
        else:
            azimuth_slope_display = "0°/min"
        
        # Couleur de la barre de confiance
        if confidence >= 0.5:
            confidence_color = "#69f0ae"  # Vert
        elif confidence >= 0.3:
            confidence_color = "#ffea00"  # Jaune
        else:
            confidence_color = "#ff4d6d"  # Rouge
    else:
        trends_box_display = "none"
        trend_label = "—"
        dist_slope_display = "—"
        dist_trend_class = ""
        dist_r2 = "—"
        azimuth_slope_display = "—"
        azimuth_r2 = "—"
        confidence_pct = 0
        confidence_color = "#4a5568"
        dist_start = "—"
        dist_end = "—"
        azimuth_start = "—"
        azimuth_end = "—"

    # Infos RANSAC (trajectoire robuste)
    ransac = analysis.get('ransac')
    if ransac:
        r2az = int(ransac.get('r2_azimut', 0) * 100)
        n_win = ransac.get('n_window', '?')
        n_tot = ransac.get('n_total_centroids', '?')
        window_info = f"{n_win}/{n_tot} centroïdes" if n_tot != n_win else f"{n_win} centroïdes"
        ransac_status = (
            f"inliers {int(ransac.get('inlier_ratio', 0) * 100)}% · "
            f"conf {int(ransac.get('confidence', 0) * 100)}% · "
            f"az R²{r2az}% · "
            f"fenêtre : {window_info}"
        )
        direction_basis = "Basé sur RANSAC (barycentres 2 min)"
    else:
        ransac_status = "—"
        direction_basis = f"Basé sur {len(analysis.get('meta_centroids', []))} méta-centroïdes"
    
    html = HTML_TEMPLATE.format(
        storm_id=analysis['storm_id'],
        airport=analysis['airport'],
        duration_min=analysis['duration_min'],
        n_strikes=analysis['n_strikes'],
        n_centroids=len(analysis['centroids']),
        n_meta=len(analysis['meta_centroids']),
        global_distance_km=analysis['global_distance_km'],
        global_speed_kmh=analysis['global_speed_kmh'],
        global_direction=analysis['global_direction'],
      ransac_status=ransac_status,
      direction_basis=direction_basis,
        radius_km=analysis.get('radius_km', 30),
        exit_status=exit_status,
        exit_box_display=exit_box_display,
        exit_time_display=exit_time_display,
        exit_time_min=exit_time_min,
        exit_dist_km=exit_dist_km,
        exit_centroid_idx=exit_centroid_idx,
        # Prédiction de sortie
        prediction_status=prediction_status,
        prediction_box_display=prediction_box_display,
        predicted_time_min=predicted_time_min,
        predicted_lat=predicted_lat,
        predicted_lon=predicted_lon,
        # Tendances (méthode hybride)
        trends_box_display=trends_box_display,
        trend_label=trend_label,
        dist_slope_display=dist_slope_display,
        dist_trend_class=dist_trend_class,
        dist_r2=dist_r2,
        azimuth_slope_display=azimuth_slope_display,
        azimuth_r2=azimuth_r2,
        confidence_pct=confidence_pct,
        confidence_color=confidence_color,
        dist_start=dist_start,
        dist_end=dist_end,
        azimuth_start=azimuth_start,
        azimuth_end=azimuth_end,
        data_json=json.dumps(analysis, default=str),
    )
    
    output_path.write_text(html, encoding='utf-8')
    print(f"  → {output_path}")


# ============================================================================
# Résumé de tous les orages
# ============================================================================

def generate_summary_html(all_analyses: list, output_path: Path) -> None:
  """Génère une page HTML résumant tous les orages analysés."""

  # Filtrer les orages sans informations utiles (vitesse nulle, pas de RANSAC, ou pas de prédiction)
  valid_analyses = [
      a for a in all_analyses
      if a.get('global_speed_kmh', 0) > 0
      and a.get('ransac') is not None
      and a['ransac'].get('inlier_ratio', 0) > 0
      and a.get('predicted_exit') is not None
  ]

  # Trier par ville puis par confiance RANSAC décroissante
  sorted_analyses = sorted(
      valid_analyses,
      key=lambda x: (x.get('airport', ''), -x['ransac'].get('confidence', 0))
  )
  displayed_analyses = sorted_analyses[:SUMMARY_MAX_ROWS]

  rows_html = ""
  for a in displayed_analyses:
    exit_info = a.get('exit_info')
    if exit_info:
      exit_time_short = exit_info['exit_time'][11:16]  # HH:MM
      exit_text = f"{exit_time_short} ({exit_info['time_to_exit_min']} min)"
      exit_color = "#ff4d6d"
    else:
      exit_text = "—"
      exit_color = "#4a5568"

    predicted_exit = a.get('predicted_exit')
    if predicted_exit:
      if predicted_exit.get('exit_time'):
        pred_time_short = predicted_exit['exit_time'][11:16]  # HH:MM
        pred_text = f"{pred_time_short} (~{predicted_exit['time_to_exit_min']} min)"
      else:
        pred_text = f"~{predicted_exit['time_to_exit_min']} min"
      pred_color = "#ffea00"
    else:
      pred_text = "—"
      pred_color = "#4a5568"

    # Colonnes RANSAC
    ransac = a.get('ransac')
    if ransac:
      r_inliers = int(ransac.get('inlier_ratio', 0) * 100)
      r_conf    = int(ransac.get('confidence', 0) * 100)
      r_azimut  = int(ransac.get('r2_azimut', 0) * 100)
      if r_inliers >= 80:
        r_color = "#69f0ae"; r_badge = ""
      elif r_inliers >= 60:
        r_color = "#ffea00"; r_badge = ""
      else:
        r_color = "#ff4d6d"; r_badge = ""
      r_conf_color = "#69f0ae" if r_conf  >= 70 else ("#ffea00" if r_conf  >= 40 else "#ff4d6d")
      r_az_color   = "#69f0ae" if r_azimut >= 70 else ("#ffea00" if r_azimut >= 40 else "#ff4d6d")
      inliers_text = f"{r_inliers}%"
      conf_text    = f"{r_conf}%"
      az_text      = f"{r_azimut}%"
    else:
      r_inliers    = 0
      r_color      = "#4a5568"
      r_conf_color = "#4a5568"
      r_az_color   = "#4a5568"
      inliers_text = "—"
      conf_text    = "—"
      az_text      = "—"

    rows_html += f'''\
    <tr onclick="window.location='{a['storm_id']}.html'" style="cursor:pointer;" data-ransac="{r_inliers}">\
      <td>{a['storm_id']}</td>\
      <td>{a['airport']}</td>\
      <td>{a['duration_min']}</td>\
      <td>{a['n_strikes']}</td>\
      <td>{len(a['centroids'])}</td>\
      <td>{a['global_distance_km']}</td>\
      <td style="color:#b2ff59;font-weight:bold;">{a['global_speed_kmh']} km/h</td>\
      <td style="color:#00e5ff;font-weight:bold;">{a['global_direction']}</td>\
      <td style="color:{r_color};font-weight:bold;">{inliers_text}</td>\
      <td style="color:{r_conf_color};font-weight:bold;">{conf_text}</td>\
      <td style="color:{r_az_color};">{az_text}</td>\
      <td style="color:{pred_color};font-weight:bold;">{pred_text}</td>\
    </tr>\
    '''

  exits_count = sum(1 for a in all_analyses if a.get('exit_info'))
  predictions_count = sum(1 for a in all_analyses if a.get('predicted_exit'))
  high_confidence = sum(1 for a in all_analyses if a.get('trends') and a['trends']['confidence'] >= 0.5)
  ransac_good = sum(1 for a in all_analyses if a.get('ransac') and a['ransac'].get('inlier_ratio', 0) >= 0.8)

  html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Storm Direction Analysis - Summary</title>
<style>
  body {{ background: #0a0c0f; color: #c8d6e5; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ color: #00e5ff; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #1e2530; white-space: nowrap; }}
  th {{
    background: #111418; color: #4a5568; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.1em;
    cursor: pointer; user-select: none;
  }}
  th:hover {{ color: #c8d6e5; }}
  th.sort-asc::after  {{ content: " ▲"; color: #00e5ff; }}
  th.sort-desc::after {{ content: " ▼"; color: #00e5ff; }}
  tr:hover {{ background: #111418; }}
  .stats {{ display: flex; gap: 32px; margin-bottom: 24px; font-size: 14px; flex-wrap: wrap; }}
  .stats span {{ color: #00e5ff; font-weight: bold; }}
  .stats .exit span {{ color: #ff4d6d; }}
  .stats .prediction span {{ color: #ffea00; }}
  .stats .confidence span {{ color: #b388ff; }}
  .stats .ransac span {{ color: #69f0ae; }}
</style>
</head>
<body>
<h1>Storm Direction Analysis</h1>
<div class="stats">
  <div>Orages analysés : <span>{len(all_analyses)}</span></div>
  <div>Affichage : <span>{len(displayed_analyses)}</span></div>
  <div class="ransac">RANSAC ≥80% inliers : <span>{ransac_good}</span></div>
  <div class="prediction">Prédictions : <span>{predictions_count}</span></div>
</div>
<p style="color:#4a5568;font-size:12px;margin-bottom:16px;">Cliquez sur un en-tête pour trier · Cliquez sur une ligne pour ouvrir la carte</p>
<table id="tbl">
  <thead><tr>
  <th>Storm ID</th>
  <th>Aéroport</th>
  <th>Durée (min)</th>
  <th>Éclairs</th>
  <th>Barycentres (2min)</th>
  <th>Distance (km)</th>
  <th>Vitesse</th>
  <th>Direction</th>
  <th>Inliers</th>
  <th>Confiance</th>
  <th>az R²</th>
  <th>Prédiction</th>
  </tr></thead>
  <tbody>
  {rows_html}
  </tbody>
</table>
<script>
(function(){{
  const tbl = document.getElementById('tbl');
  const ths = Array.from(tbl.querySelectorAll('thead th'));
  let lastCol = -1, asc = true;
  ths.forEach((th, ci) => {{
    th.title = 'Trier par ' + th.textContent.trim();
    th.addEventListener('click', () => {{
      asc = (lastCol === ci) ? !asc : true;
      lastCol = ci;
      ths.forEach(h => h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(asc ? 'sort-asc' : 'sort-desc');
      const tbody = tbl.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {{
        const va = a.cells[ci].textContent.trim();
        const vb = b.cells[ci].textContent.trim();
        const na = parseFloat(va); const nb = parseFloat(vb);
        const cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb : va.localeCompare(vb, 'fr');
        return asc ? cmp : -cmp;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}})();
</script>
</body>
</html>
'''

  output_path.write_text(html, encoding='utf-8')
  print(f"\nResume : {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse de direction des orages")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Chemin vers le CSV")
    parser.add_argument("--storm_id", type=str, default=None, help="Analyser un seul orage")
    parser.add_argument("--min_duration", type=float, default=MIN_DURATION_MINUTES, help="Durée minimale (min)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Dossier de sortie")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'orages analysés")
    parser.add_argument("--airport", type=str, default=None, help="Filtrer par aéroport (ex: Ajaccio)")
    parser.add_argument("--limit_per_airport", type=int, default=None, help="Prendre N orages par aéroport")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Chargement de {args.csv}...")
    df = pd.read_csv(args.csv, parse_dates=['date'])
    
    # Filtrer les orages avec durée > min_duration
    if 'duration_min' in df.columns:
        valid_storms = df.groupby('storm_id').first()
        valid_storms = valid_storms[valid_storms['duration_min'] >= args.min_duration]
        storm_ids = valid_storms.index.tolist()
    else:
        # Calculer la durée si non présente
        storm_durations = df.groupby('storm_id')['date'].agg(['min', 'max'])
        storm_durations['duration'] = (storm_durations['max'] - storm_durations['min']).dt.total_seconds() / 60
        storm_ids = storm_durations[storm_durations['duration'] >= args.min_duration].index.tolist()
    
    if args.storm_id:
        storm_ids = [args.storm_id]

    if args.airport:
        airport_storms = df[df['airport'] == args.airport]['storm_id'].dropna().unique().tolist()
        storm_ids = [s for s in storm_ids if s in airport_storms]
        print(f"Filtre aeroport '{args.airport}' : {len(storm_ids)} orages")

    if args.limit_per_airport:
        storm_to_airport = df.dropna(subset=['storm_id']).groupby('storm_id')['airport'].first().to_dict()
        per_airport: dict = {}
        selected = []
        for sid in storm_ids:
            ap = storm_to_airport.get(sid, 'unknown')
            if per_airport.get(ap, 0) < args.limit_per_airport:
                selected.append(sid)
                per_airport[ap] = per_airport.get(ap, 0) + 1
        storm_ids = selected
        print(f"Limite par aeroport ({args.limit_per_airport}) : {len(storm_ids)} orages au total")
        for ap, n in sorted(per_airport.items()):
            print(f"   {ap}: {n}")

    if args.limit:
        storm_ids = storm_ids[:args.limit]
    
    print(f"{len(storm_ids)} orages a analyser (duree >= {args.min_duration} min)\n")
    
    all_analyses = []
    
    for storm_id in storm_ids:
        df_storm = df[df['storm_id'] == storm_id]
        
        if len(df_storm) < 3:
            print(f"  ⏭️  {storm_id}: pas assez d'éclairs ({len(df_storm)})")
            continue
        
        analysis = analyze_storm(df_storm, storm_id)
        
        if analysis is None or len(analysis['centroids']) < 2:
            print(f"  ⏭️  {storm_id}: pas assez de centroïdes")
            continue
        
        all_analyses.append(analysis)
        
        # Générer HTML individuel
        html_path = output_dir / f"{storm_id}.html"
        generate_html(analysis, html_path)
        
        print(f"  ✓ {storm_id}: {analysis['global_speed_kmh']} km/h → {analysis['global_direction']} ({analysis['global_distance_km']} km)")
    
    # Générer résumé
    if all_analyses:
        generate_summary_html(all_analyses, output_dir / "index.html")
        
        # Sauvegarder aussi en JSON
        json_path = output_dir / "storm_directions.json"
        with open(json_path, 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)
        print(f"JSON : {json_path}")
    
    print(f"\nTermine ! {len(all_analyses)} orages analyses.")
    print(f"   Ouvre {output_dir / 'index.html'} pour voir le résumé.")


if __name__ == "__main__":
    main()
