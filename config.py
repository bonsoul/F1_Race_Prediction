"""
config.py — central configuration for the F1 prediction platform
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR  = DATA_DIR / "features"
MODELS_DIR    = ROOT / "models" / "saved"

for p in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = DATA_DIR / "f1_platform.duckdb"
FASTF1_CACHE = DATA_DIR / "fastf1_cache"
FASTF1_CACHE.mkdir(parents=True, exist_ok=True)

# ── Data sources ───────────────────────────────────────────────────────────
ERGAST_BASE   = "https://ergast.com/api/f1"
OPENF1_BASE   = "https://api.openf1.org/v1"

# ── Seasons ────────────────────────────────────────────────────────────────
HISTORICAL_SEASONS = list(range(2021, 2026))   # training data
CURRENT_SEASON     = 2026
TRAIN_SEASONS      = list(range(2021, 2025))   # strict temporal split
VALIDATION_SEASON  = 2025

# ── Feature lists ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "grid_position",
    "quali_gap_to_pole_s",
    "driver_rolling_elo",
    "driver_rolling_points_5r",
    "driver_rolling_dnf_rate_5r",
    "constructor_rolling_points_5r",
    "constructor_pace_rank",
    "constructor_reliability_5r",
    "track_overtaking_index",
    "is_wet_race",
    "air_temp_c",
    "track_temp_c",
    "lap1_sector1_gap_s",       # race-week feature (OpenF1 / FastF1)
    "quali_position",
    "season_race_number",
]

CATEGORICAL_FEATURES = [
    "circuit_type",             # street | hybrid | highspeed | technical
    "tyre_compound_start",
]

TARGET_TOP3   = "is_top3"          # binary
TARGET_WINNER = "is_winner"        # binary (use as ranking signal)
TARGET_POINTS = "points_scored"    # regression

# ── Model params ───────────────────────────────────────────────────────────
XGB_TOP3_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

XGB_POINTS_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

LGBM_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
}

# ── Circuit type mapping ──────────────────────────────────────────────────
CIRCUIT_TYPES = {
    "monaco":        "street",
    "singapore":     "street",
    "baku":          "street",
    "jeddah":        "street",
    "las_vegas":     "street",
    "miami":         "street",
    "monza":         "highspeed",
    "spa":           "highspeed",
    "silverstone":   "highspeed",
    "suzuka":        "technical",
    "hungaroring":   "technical",
    "barcelona":     "hybrid",
    "zandvoort":     "hybrid",
    # default → "hybrid"
}

# ── ELO settings ──────────────────────────────────────────────────────────
ELO_K_FACTOR    = 32
ELO_BASE_RATING = 1500
