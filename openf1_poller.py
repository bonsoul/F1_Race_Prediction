"""
ingestion/openf1_poller.py
Polls the OpenF1 API during race weekends for near-real-time data:
  - Session info  (which sessions are live/upcoming)
  - Driver info   (number → code mapping)
  - Lap data      (pace, sector times)
  - Car data      (speed, throttle, brake — telemetry)
  - Pit data      (live pit stops)
  - Weather       (real-time track conditions)

Can be run as a one-off pull or scheduled during a race weekend.

Usage:
    # Pull latest session data right now
    python -m ingestion.openf1_poller --mode pull

    # Schedule polling every 60s during a live race weekend
    python -m ingestion.openf1_poller --mode schedule --interval 60
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENF1_BASE, RAW_DIR

OPENF1_OUT = RAW_DIR / "openf1"
OPENF1_OUT.mkdir(parents=True, exist_ok=True)


# ── HTTP helper ────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> list[dict]:
    url = f"{OPENF1_BASE}/{endpoint}"
    try:
        r = httpx.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        logger.warning(f"OpenF1 request failed ({exc}): {url}")
        return []


# ── Endpoint pullers ───────────────────────────────────────────────────────

def get_sessions(year: int | None = None) -> pd.DataFrame:
    params = {}
    if year:
        params["year"] = year
    data = _get("sessions", params)
    return pd.DataFrame(data) if data else pd.DataFrame()


def get_drivers(session_key: int) -> pd.DataFrame:
    data = _get("drivers", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


def get_laps(session_key: int, driver_number: int | None = None) -> pd.DataFrame:
    params: dict = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    data = _get("laps", params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Convert ISO timestamps to datetime
    for col in ["date_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def get_car_data(session_key: int, driver_number: int,
                  max_samples: int = 500) -> pd.DataFrame:
    """Telemetry — speed, throttle, brake, DRS per sample."""
    data = _get("car_data", {
        "session_key": session_key,
        "driver_number": driver_number,
    })
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data[:max_samples])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def get_pit_stops(session_key: int) -> pd.DataFrame:
    data = _get("pit", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


def get_weather(session_key: int) -> pd.DataFrame:
    data = _get("weather", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def get_race_control(session_key: int) -> pd.DataFrame:
    """Safety car, virtual SC, red flag messages."""
    data = _get("race_control", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


def get_intervals(session_key: int) -> pd.DataFrame:
    """Live gap to leader and gap ahead per driver."""
    data = _get("intervals", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


# ── Live session discovery ─────────────────────────────────────────────────

def find_latest_session(session_type: str = "Race") -> dict | None:
    """Return the most recent session of the given type."""
    sessions = get_sessions()
    if sessions.empty:
        return None
    races = sessions[sessions["session_name"] == session_type].copy()
    if races.empty:
        return None
    races = races.sort_values("date_start", ascending=False)
    return races.iloc[0].to_dict()


# ── One-off pull ───────────────────────────────────────────────────────────

def pull_latest(session_type: str = "Race") -> None:
    session = find_latest_session(session_type)
    if not session:
        logger.warning("No recent session found on OpenF1.")
        return

    sk  = int(session["session_key"])
    tag = f"{session.get('year', 'unk')}_{session.get('circuit_short_name', 'unk')}_{session_type.lower()}"
    out = OPENF1_OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pulling OpenF1 data → session_key={sk} ({tag})")

    # Session metadata
    pd.DataFrame([session]).to_parquet(out / "session.parquet", index=False)

    # Drivers
    drivers = get_drivers(sk)
    if not drivers.empty:
        drivers.to_parquet(out / "drivers.parquet", index=False)
        logger.info(f"  drivers: {len(drivers)} rows")

    # Lap data
    laps = get_laps(sk)
    if not laps.empty:
        laps.to_parquet(out / "laps.parquet", index=False)
        logger.info(f"  laps: {len(laps)} rows")

    # Pit stops
    pits = get_pit_stops(sk)
    if not pits.empty:
        pits.to_parquet(out / "pit_stops.parquet", index=False)
        logger.info(f"  pit_stops: {len(pits)} rows")

    # Weather
    weather = get_weather(sk)
    if not weather.empty:
        weather.to_parquet(out / "weather.parquet", index=False)
        logger.info(f"  weather: {len(weather)} rows")

    # Race control messages
    rc = get_race_control(sk)
    if not rc.empty:
        rc.to_parquet(out / "race_control.parquet", index=False)
        logger.info(f"  race_control: {len(rc)} rows")

    # Intervals (gap to leader)
    intervals = get_intervals(sk)
    if not intervals.empty:
        intervals.to_parquet(out / "intervals.parquet", index=False)
        logger.info(f"  intervals: {len(intervals)} rows")

    # Compute pace summary from laps (mean lap time per driver, excl. pit laps)
    if not laps.empty and "lap_duration" in laps.columns and "driver_number" in laps.columns:
        pace = (
            laps[laps["lap_duration"] < laps["lap_duration"].quantile(0.95)]
            .groupby("driver_number")["lap_duration"]
            .agg(["mean", "median", "min", "count"])
            .rename(columns={"mean": "avg_lap_s", "median": "median_lap_s",
                             "min": "fastest_lap_s", "count": "laps_counted"})
            .reset_index()
        )
        pace["session_key"] = sk
        pace.to_parquet(out / "pace_summary.parquet", index=False)
        logger.info(f"  pace_summary: {len(pace)} drivers")

    logger.success(f"OpenF1 pull complete → {out}")


# ── Live scheduler ─────────────────────────────────────────────────────────

def schedule_polling(interval_seconds: int = 60) -> None:
    logger.info(f"Starting OpenF1 poller — interval {interval_seconds}s. Ctrl-C to stop.")
    scheduler = BlockingScheduler(timezone=timezone.utc)
    scheduler.add_job(pull_latest, "interval", seconds=interval_seconds,
                      kwargs={"session_type": "Race"})
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Poller stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pull", "schedule"], default="pull")
    parser.add_argument("--interval", type=int, default=60,
                        help="Polling interval in seconds (schedule mode)")
    parser.add_argument("--session-type", default="Race",
                        choices=["Race", "Qualifying", "Practice 1",
                                  "Practice 2", "Practice 3", "Sprint"])
    args = parser.parse_args()

    if args.mode == "pull":
        pull_latest(args.session_type)
    else:
        schedule_polling(args.interval)
