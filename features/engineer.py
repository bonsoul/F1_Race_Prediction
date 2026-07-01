"""
features/engineer.py
Builds the final modelling feature table from the DuckDB warehouse.

Features computed here:
  - Rolling driver ELO rating
  - Rolling driver points/DNF rate (last 5 races)
  - Rolling constructor points/reliability (last 5 races)
  - Constructor pace rank (circuit-type adjusted)
  - Track overtaking index
  - Circuit type encoding

Output:
    data/features/model_features.parquet   ← clean ML-ready table
    data/features/feature_meta.json        ← column metadata

Usage:
    python -m features.engineer
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DUCKDB_PATH, FEATURES_DIR, CIRCUIT_TYPES,
    ELO_K_FACTOR, ELO_BASE_RATING,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_TOP3, TARGET_WINNER, TARGET_POINTS,
)
from utils.warehouse import query


# ── ELO computation ────────────────────────────────────────────────────────

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def compute_driver_elo(results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a rolling ELO rating for each driver based on head-to-head
    finish position comparisons across all races (chronological order).
    Returns a DataFrame with (season, round, driver_id, driver_elo).
    """
    results = results.sort_values(["season", "round", "finish_position"]).copy()
    ratings: dict[str, float] = {}
    elo_rows = []

    for (season, round_), grp in results.groupby(["season", "round"]):
        grp = grp.dropna(subset=["finish_position"]).copy()
        drivers = grp["driver_id"].tolist()

        # Snapshot ratings BEFORE updating (use pre-race ELO as feature)
        for drv in drivers:
            ratings.setdefault(drv, ELO_BASE_RATING)
            elo_rows.append({
                "season":     season,
                "round":      round_,
                "driver_id":  drv,
                "driver_elo": ratings[drv],
            })

        # Pairwise ELO update: all (i, j) pairs where i finished ahead of j
        for i, row_i in grp.iterrows():
            for j, row_j in grp.iterrows():
                if row_i["finish_position"] >= row_j["finish_position"]:
                    continue
                drv_a = row_i["driver_id"]
                drv_b = row_j["driver_id"]
                ea = _expected(ratings[drv_a], ratings[drv_b])
                ratings[drv_a] += ELO_K_FACTOR * (1 - ea)
                ratings[drv_b] += ELO_K_FACTOR * (0 - (1 - ea))

    return pd.DataFrame(elo_rows)


# ── Rolling window features ────────────────────────────────────────────────

def rolling_driver_features(results: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Per driver, rolling over the last `window` races (before current):
      - rolling_points_5r     : mean points scored
      - rolling_dnf_rate_5r   : fraction of races that were DNF/retirement
    """
    results = results.sort_values(["driver_id", "season", "round"]).copy()
    results["is_dnf"] = (~results["is_classified"]).astype(float)

    results["driver_rolling_points_5r"] = (
        results.groupby("driver_id")["points_scored"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    results["driver_rolling_dnf_rate_5r"] = (
        results.groupby("driver_id")["is_dnf"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return results[["season", "round", "driver_id",
                     "driver_rolling_points_5r", "driver_rolling_dnf_rate_5r"]]


def rolling_constructor_features(results: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Per constructor per race, rolling over last `window` races:
      - constructor_rolling_points_5r  : mean points (team total per race)
      - constructor_reliability_5r     : 1 - mean DNF rate
    """
    team = (
        results.groupby(["season", "round", "constructor_id"])
        .agg(
            team_points=("points_scored", "sum"),
            team_dnfs=("is_classified", lambda x: (~x).sum()),
            team_cars=("driver_id", "count"),
        )
        .reset_index()
    )
    team = team.sort_values(["constructor_id", "season", "round"])
    team["dnf_rate"] = team["team_dnfs"] / team["team_cars"]

    team["constructor_rolling_points_5r"] = (
        team.groupby("constructor_id")["team_points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    team["constructor_reliability_5r"] = 1 - (
        team.groupby("constructor_id")["dnf_rate"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return team[["season", "round", "constructor_id",
                  "constructor_rolling_points_5r", "constructor_reliability_5r"]]


def constructor_pace_rank(results: pd.DataFrame) -> pd.DataFrame:
    """
    Constructor pace rank: for each race, rank constructors by mean
    grid position of their drivers (lower grid = better pace).
    Returns integer rank (1 = best pace).
    """
    pace = (
        results.groupby(["season", "round", "constructor_id"])["grid_position"]
        .mean()
        .reset_index()
        .rename(columns={"grid_position": "mean_grid"})
    )
    pace["constructor_pace_rank"] = (
        pace.groupby(["season", "round"])["mean_grid"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return pace[["season", "round", "constructor_id", "constructor_pace_rank"]]


# ── Track features ─────────────────────────────────────────────────────────

# Overtaking difficulty 0–1 (higher = harder to overtake → grid matters more)
OVERTAKING_INDEX = {
    "street":    0.90,
    "technical": 0.65,
    "hybrid":    0.50,
    "highspeed": 0.25,
}


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    df["circuit_type"] = df["circuit_id"].map(
        lambda c: CIRCUIT_TYPES.get(c, "hybrid")
    )
    df["track_overtaking_index"] = df["circuit_type"].map(OVERTAKING_INDEX)
    return df


# ── Quali gap to pole normalisation ───────────────────────────────────────

def add_quali_gap(df: pd.DataFrame) -> pd.DataFrame:
    pole_times = (
        df.groupby(["season", "round"])["best_quali_s"]
        .min()
        .rename("pole_time_s")
        .reset_index()
    )
    df = df.merge(pole_times, on=["season", "round"], how="left")
    df["quali_gap_to_pole_s"] = (df["best_quali_s"] - df["pole_time_s"]).clip(lower=0)
    return df


# ── Master build ───────────────────────────────────────────────────────────

def build_features() -> pd.DataFrame:
    logger.info("Loading base view from warehouse …")
    base = query("SELECT * FROM v_race_features")
    logger.info(f"  Base rows: {len(base)}")

    # Load full results for rolling computations
    results_full = query("""
        SELECT season, round, driver_id, constructor_id,
               points_scored, is_classified, finish_position, grid_position
        FROM race_results
    """)

    # ── ELO ──
    logger.info("Computing driver ELO …")
    elo = compute_driver_elo(results_full)
    base = base.merge(elo, on=["season", "round", "driver_id"], how="left")

    # ── Rolling driver features ──
    logger.info("Computing rolling driver features …")
    drv_roll = rolling_driver_features(results_full)
    base = base.merge(drv_roll, on=["season", "round", "driver_id"], how="left")

    # ── Rolling constructor features ──
    logger.info("Computing rolling constructor features …")
    con_roll = rolling_constructor_features(results_full)
    base = base.merge(con_roll, on=["season", "round", "constructor_id"], how="left")

    # ── Constructor pace rank ──
    logger.info("Computing constructor pace rank …")
    pace = constructor_pace_rank(results_full)
    base = base.merge(pace, on=["season", "round", "constructor_id"], how="left")

    # ── Circuit features ──
    logger.info("Adding circuit features …")
    base = add_circuit_features(base)

    # ── Quali gap to pole ──
    logger.info("Computing quali gap to pole …")
    if "best_quali_s" in base.columns:
        base = add_quali_gap(base)
    else:
        base["quali_gap_to_pole_s"] = np.nan

    # ── Season race number ──
    base["season_race_number"] = base.groupby("season")["round"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )

    # ── Defaults for race-week-only features ──
    if "lap1_sector1_gap_s" not in base.columns:
        base["lap1_sector1_gap_s"] = np.nan

    # ── Targets ──
    base[TARGET_TOP3]   = base["is_top3"].astype(int)
    base[TARGET_WINNER] = base["is_winner"].astype(int)
    base[TARGET_POINTS] = base["points_scored"]

    # ── Encode categoricals ──
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in base.columns:
            base[col] = le.fit_transform(base[col].fillna("unknown"))

    # ── Final column selection ──
    id_cols   = ["season", "round", "circuit_id", "driver_id",
                  "driver_code", "constructor_id"]
    target_cols = [TARGET_TOP3, TARGET_WINNER, TARGET_POINTS]
    feat_cols   = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    out = base[id_cols + feat_cols + target_cols].copy()

    # Impute remaining NaNs with column medians
    for col in feat_cols:
        if col in out.columns:
            med = out[col].median()
            out[col] = out[col].fillna(med)

    logger.success(f"Feature table: {out.shape[0]} rows × {out.shape[1]} cols")

    # Save
    out_path = FEATURES_DIR / "model_features.parquet"
    out.to_parquet(out_path, index=False)
    logger.success(f"Saved → {out_path}")

    # Metadata
    meta = {
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "targets":               target_cols,
        "n_rows":                len(out),
        "seasons":               sorted(out["season"].unique().tolist()),
    }
    with open(FEATURES_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out


if __name__ == "__main__":
    build_features()
