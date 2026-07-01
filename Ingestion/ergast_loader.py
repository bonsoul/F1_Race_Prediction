"""
Ingestion/ergast_loader.py
Batch-loads multi-season F1 data from the Ergast API into Parquet files.

Usage:
    python -m Ingestion.ergast_loader --seasons 2021 2022 2023 2024 2025
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ERGAST_BASE, RAW_DIR


# â”€â”€ HTTP helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET from Ergast with retry + back-off. Returns the MRData payload."""
    url = f"{ERGAST_BASE}/{endpoint}.json"
    params = {**(params or {}), "limit": 1000}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()["MRData"]
        except requests.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(f"Ergast error ({exc}), retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Ergast request failed after {retries} retries: {url}")

def _load_cached_frame(cache_path: Path, label: str) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        logger.info(f"[cache] Using cached {label} from {cache_path}")
        return pd.read_parquet(cache_path)
    except Exception as exc:
        logger.warning(f"[cache] Could not read cached {label} from {cache_path}: {exc}")
        return None


def _fetch_or_cached_frame(
    season: int,
    label: str,
    fetcher,
    cache_path: Path,
    *args,
    prefer_cache: bool = False,
):
    if prefer_cache:
        cached = _load_cached_frame(cache_path, label)
        if cached is not None:
            return cached, True

    try:
        frame = fetcher(*args)
    except RuntimeError as exc:
        logger.warning(f"[{season}] {label} unavailable: {exc}")
        frame = None

    if frame is None or frame.empty:
        cached = _load_cached_frame(cache_path, label)
        if cached is not None:
            return cached, True

    return frame, False


# â”€â”€ Race schedule â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_schedule(season: int) -> pd.DataFrame:
    data = _get(f"{season}")
    races = data["RaceTable"]["Races"]
    rows = []
    for r in races:
        rows.append({
            "season":       int(r["season"]),
            "round":        int(r["round"]),
            "race_name":    r["raceName"],
            "circuit_id":   r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "country":      r["Circuit"]["Location"]["country"],
            "date":         r["date"],
            "time":         r.get("time", None),
        })
    return pd.DataFrame(rows)


# â”€â”€ Race results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_results(season: int) -> pd.DataFrame:
    data = _get(f"{season}/results")
    races = data["RaceTable"]["Races"]
    rows = []
    for r in races:
        for res in r["Results"]:
            status = res["status"]
            rows.append({
                "season":         int(r["season"]),
                "round":          int(r["round"]),
                "circuit_id":     r["Circuit"]["circuitId"],
                "driver_id":      res["Driver"]["driverId"],
                "driver_code":    res["Driver"].get("code", ""),
                "constructor_id": res["Constructor"]["constructorId"],
                "grid_position":  int(res.get("grid", 0)),
                "finish_position":int(res["position"]),
                "points_scored":  float(res["points"]),
                "laps_completed": int(res["laps"]),
                "status":         status,
                "is_classified":  "Finished" in status or "Lap" in status,
                "fastest_lap_rank": int(res.get("FastestLap", {}).get("rank", 0)),
                "is_winner":      res["position"] == "1",
                "is_top3":        int(res["position"]) <= 3,
            })
    return pd.DataFrame(rows)


# â”€â”€ Qualifying results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_qualifying(season: int) -> pd.DataFrame:
    data = _get(f"{season}/qualifying")
    races = data["RaceTable"]["Races"]
    rows = []
    for r in races:
        for q in r.get("QualifyingResults", []):
            def _to_sec(t: str) -> float | None:
                """'1:23.456' â†’ 83.456"""
                if not t:
                    return None
                try:
                    parts = t.split(":")
                    return float(parts[0]) * 60 + float(parts[1])
                except Exception:
                    return None

            q1 = _to_sec(q.get("Q1", ""))
            q2 = _to_sec(q.get("Q2", ""))
            q3 = _to_sec(q.get("Q3", ""))
            best = next((x for x in [q3, q2, q1] if x is not None), None)
            rows.append({
                "season":          int(r["season"]),
                "round":           int(r["round"]),
                "circuit_id":      r["Circuit"]["circuitId"],
                "driver_id":       q["Driver"]["driverId"],
                "constructor_id":  q["Constructor"]["constructorId"],
                "quali_position":  int(q["position"]),
                "q1_time_s":       q1,
                "q2_time_s":       q2,
                "q3_time_s":       q3,
                "best_quali_s":    best,
            })
    return pd.DataFrame(rows)


# â”€â”€ Driver standings (per round) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_driver_standings(season: int, n_rounds: int) -> pd.DataFrame:
    rows = []
    for rnd in range(1, n_rounds + 1):
        data = _get(f"{season}/{rnd}/driverStandings")
        standings = data["StandingsTable"]["StandingsLists"]
        if not standings:
            continue
        for s in standings[0]["DriverStandings"]:
            rows.append({
                "season":     season,
                "round":      rnd,
                "driver_id":  s["Driver"]["driverId"],
                "position":   int(s["position"]),
                "points":     float(s["points"]),
                "wins":       int(s["wins"]),
            })
    return pd.DataFrame(rows)


# â”€â”€ Constructor standings (per round) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_constructor_standings(season: int, n_rounds: int) -> pd.DataFrame:
    rows = []
    for rnd in range(1, n_rounds + 1):
        data = _get(f"{season}/{rnd}/constructorStandings")
        standings = data["StandingsTable"]["StandingsLists"]
        if not standings:
            continue
        for s in standings[0]["ConstructorStandings"]:
            rows.append({
                "season":          season,
                "round":           rnd,
                "constructor_id":  s["Constructor"]["constructorId"],
                "position":        int(s["position"]),
                "points":          float(s["points"]),
                "wins":            int(s["wins"]),
            })
    return pd.DataFrame(rows)


# â”€â”€ Lap times â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_lap_times(season: int, round_num: int) -> pd.DataFrame:
    """Ergast lap times endpoint â€” large payload, use sparingly."""
    rows = []
    offset = 0
    while True:
        data = _get(f"{season}/{round_num}/laps", {"offset": offset})
        laps_table = data["RaceTable"]["Races"]
        if not laps_table:
            break
        for lap in laps_table[0]["Laps"]:
            lap_num = int(lap["number"])
            for timing in lap["Timings"]:
                def _lap_to_sec(t: str) -> float | None:
                    try:
                        p = t.split(":")
                        return float(p[0]) * 60 + float(p[1])
                    except Exception:
                        return None
                rows.append({
                    "season":    season,
                    "round":     round_num,
                    "lap":       lap_num,
                    "driver_id": timing["driverId"],
                    "position":  int(timing["position"]),
                    "lap_time_s": _lap_to_sec(timing["time"]),
                })
        total = int(data.get("total", 0))
        offset += int(data.get("limit", 1000))
        if offset >= total:
            break
    return pd.DataFrame(rows)


# â”€â”€ Pit stops â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_pit_stops(season: int, round_num: int) -> pd.DataFrame:
    data = _get(f"{season}/{round_num}/pitstops")
    races = data["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for pit in races[0].get("PitStops", []):
        def _dur(d: str) -> float | None:
            try:
                return float(d)
            except Exception:
                return None
        rows.append({
            "season":       season,
            "round":        round_num,
            "driver_id":    pit["driverId"],
            "stop_number":  int(pit["stop"]),
            "lap":          int(pit["lap"]),
            "duration_s":   _dur(pit.get("duration", "")),
        })
    return pd.DataFrame(rows)


# â”€â”€ Main orchestrator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_season(season: int) -> bool:
    out = RAW_DIR / str(season)
    out.mkdir(parents=True, exist_ok=True)

    schedule_cache = out / "schedule.parquet"
    prefer_cache = schedule_cache.exists()

    logger.info(f"[{season}] Fetching schedule ...")
    schedule, schedule_from_cache = _fetch_or_cached_frame(
        season, "schedule", fetch_schedule, schedule_cache, season, prefer_cache=prefer_cache
    )
    if schedule is None or schedule.empty:
        logger.warning(
            f"[{season}] Skipping Ergast season because the schedule could not be fetched."
        )
        return False
    schedule.to_parquet(schedule_cache, index=False)
    n_rounds = len(schedule)
    prefer_cache = prefer_cache or schedule_from_cache

    logger.info(f"[{season}] Fetching results ({n_rounds} rounds) ...")
    results, _ = _fetch_or_cached_frame(
        season,
        "results",
        fetch_results,
        out / "results.parquet",
        season,
        prefer_cache=prefer_cache,
    )
    if results is not None and not results.empty:
        results.to_parquet(out / "results.parquet", index=False)

    logger.info(f"[{season}] Fetching qualifying ...")
    quali, _ = _fetch_or_cached_frame(
        season,
        "qualifying",
        fetch_qualifying,
        out / "qualifying.parquet",
        season,
        prefer_cache=prefer_cache,
    )
    if quali is not None and not quali.empty:
        quali.to_parquet(out / "qualifying.parquet", index=False)

    logger.info(f"[{season}] Fetching driver standings ...")
    driver_standings, _ = _fetch_or_cached_frame(
        season,
        "driver standings",
        fetch_driver_standings,
        out / "driver_standings.parquet",
        season,
        n_rounds,
        prefer_cache=prefer_cache,
    )
    if driver_standings is not None and not driver_standings.empty:
        driver_standings.to_parquet(out / "driver_standings.parquet", index=False)

    logger.info(f"[{season}] Fetching constructor standings ...")
    constructor_standings, _ = _fetch_or_cached_frame(
        season,
        "constructor standings",
        fetch_constructor_standings,
        out / "constructor_standings.parquet",
        season,
        n_rounds,
        prefer_cache=prefer_cache,
    )
    if constructor_standings is not None and not constructor_standings.empty:
        constructor_standings.to_parquet(out / "constructor_standings.parquet", index=False)

    logger.info(f"[{season}] Fetching pit stops ...")
    pit_cache = out / "pit_stops.parquet"
    if prefer_cache:
        cached_pit = _load_cached_frame(pit_cache, "pit stops")
        if cached_pit is not None:
            cached_pit.to_parquet(pit_cache, index=False)
            logger.success(f"[{season}] All Ergast data saved to {out}")
            return True

    pit_rows = []
    pit_failed = False
    for rnd in tqdm(range(1, n_rounds + 1), desc=f"{season} pit stops"):
        try:
            df = fetch_pit_stops(season, rnd)
            if not df.empty:
                pit_rows.append(df)
            time.sleep(0.3)   # polite rate-limit
        except Exception as exc:
            pit_failed = True
            logger.warning(f"  Round {rnd} pit stop failed: {exc}")

    if pit_failed:
        cached_pit = _load_cached_frame(pit_cache, "pit stops")
        if cached_pit is not None:
            cached_pit.to_parquet(pit_cache, index=False)
            logger.success(f"[{season}] All Ergast data saved to {out}")
            return True

    if pit_rows:
        pd.concat(pit_rows, ignore_index=True).to_parquet(
            pit_cache, index=False
        )
    else:
        logger.warning(f"[{season}] No pit stop data available.")

    logger.success(f"[{season}] All Ergast data saved to {out}")
    return True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()
    for s in args.seasons:
        run_season(s)
