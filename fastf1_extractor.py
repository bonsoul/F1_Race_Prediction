"""
ingestion/fastf1_extractor.py
Extracts structured session-level data (quali laps, race laps, telemetry)
using the FastF1 Python library with local caching.

Usage:
    python -m ingestion.fastf1_extractor --season 2025 --rounds 1 5
"""
import argparse
from pathlib import Path

import fastf1
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, FASTF1_CACHE

fastf1.Cache.enable_cache(str(FASTF1_CACHE))


# ── Lap data ──────────────────────────────────────────────────────────────

def extract_race_laps(season: int, round_num: int) -> pd.DataFrame:
    """Full lap-by-lap timing for a race session."""
    logger.info(f"  FastF1 race laps {season} R{round_num}")
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        logger.warning(f"  Could not load race session: {exc}")
        return pd.DataFrame()

    laps = session.laps[
        ["DriverNumber", "Driver", "LapNumber", "LapTime",
         "Sector1Time", "Sector2Time", "Sector3Time",
         "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
         "Compound", "TyreLife", "TrackStatus", "IsAccurate"]
    ].copy()

    # Convert timedelta columns to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col}_s"] = laps[col].dt.total_seconds()
    laps.drop(columns=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"],
              inplace=True)

    laps["season"] = season
    laps["round"]  = round_num
    return laps.reset_index(drop=True)


def extract_quali_laps(season: int, round_num: int) -> pd.DataFrame:
    """Qualifying lap times including Q1/Q2/Q3 best sectors."""
    logger.info(f"  FastF1 quali laps {season} R{round_num}")
    try:
        session = fastf1.get_session(season, round_num, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        logger.warning(f"  Could not load quali session: {exc}")
        return pd.DataFrame()

    laps = session.laps.pick_accurate()
    best = (
        laps.groupby("Driver")
        .apply(lambda g: g.loc[g["LapTime"].idxmin()])
        .reset_index(drop=True)
    )
    keep = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "SpeedI1", "SpeedFL", "Compound"]
    best = best[[c for c in keep if c in best.columns]].copy()
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in best.columns:
            best[f"{col}_s"] = best[col].dt.total_seconds()
            best.drop(columns=[col], inplace=True)
    best["season"] = season
    best["round"]  = round_num
    return best.reset_index(drop=True)


def extract_weather(season: int, round_num: int, session_type: str = "R") -> pd.DataFrame:
    """Race/quali weather snapshot — air temp, track temp, rainfall."""
    logger.info(f"  FastF1 weather {season} R{round_num} {session_type}")
    try:
        session = fastf1.get_session(season, round_num, session_type)
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        logger.warning(f"  Could not load weather: {exc}")
        return pd.DataFrame()

    w = session.weather_data
    if w is None or w.empty:
        return pd.DataFrame()

    summary = pd.DataFrame([{
        "season":       season,
        "round":        round_num,
        "session_type": session_type,
        "air_temp_c":   w["AirTemp"].mean(),
        "track_temp_c": w["TrackTemp"].mean(),
        "humidity":     w["Humidity"].mean(),
        "wind_speed":   w["WindSpeed"].mean(),
        "is_wet_race":  int(w["Rainfall"].any()),
    }])
    return summary


def extract_sector1_gap(season: int, round_num: int) -> pd.DataFrame:
    """
    Race lap-1 sector-1 gap to leader — a strong race-week predictor.
    Returns gap in seconds per driver for lap 1 only.
    """
    logger.info(f"  FastF1 lap-1 S1 gaps {season} R{round_num}")
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        logger.warning(f"  Could not load session: {exc}")
        return pd.DataFrame()

    lap1 = session.laps[session.laps["LapNumber"] == 1].copy()
    if lap1.empty:
        return pd.DataFrame()

    lap1["Sector1Time_s"] = lap1["Sector1Time"].dt.total_seconds()
    best_s1 = lap1["Sector1Time_s"].min()
    lap1["lap1_sector1_gap_s"] = lap1["Sector1Time_s"] - best_s1
    out = lap1[["Driver", "lap1_sector1_gap_s"]].copy()
    out["season"] = season
    out["round"]  = round_num
    return out.reset_index(drop=True)


# ── Main orchestrator ─────────────────────────────────────────────────────

def run(season: int, rounds: list[int]) -> None:
    out = RAW_DIR / str(season) / "fastf1"
    out.mkdir(parents=True, exist_ok=True)

    all_race_laps   = []
    all_quali_laps  = []
    all_weather     = []
    all_s1_gaps     = []

    for rnd in rounds:
        all_race_laps.append(extract_race_laps(season, rnd))
        all_quali_laps.append(extract_quali_laps(season, rnd))
        all_weather.append(extract_weather(season, rnd, "R"))
        all_s1_gaps.append(extract_sector1_gap(season, rnd))

    def _save(frames, name):
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if not df.empty:
            df.to_parquet(out / f"{name}.parquet", index=False)
            logger.success(f"  Saved {name}.parquet ({len(df)} rows)")
        else:
            logger.warning(f"  No data for {name}")

    _save(all_race_laps,  "race_laps")
    _save(all_quali_laps, "quali_laps")
    _save(all_weather,    "weather")
    _save(all_s1_gaps,    "lap1_s1_gaps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--rounds", nargs="+", type=int, default=list(range(1, 25)))
    args = parser.parse_args()
    run(args.season, args.rounds)
