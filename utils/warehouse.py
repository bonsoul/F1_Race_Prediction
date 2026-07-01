"""
utils/warehouse.py
Loads all Parquet files into a DuckDB warehouse for fast SQL-based
feature engineering. Creates normalised tables and key joined views.

Usage:
    python -m utils.warehouse --build
    python -m utils.warehouse --query "SELECT * FROM race_results LIMIT 5"
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, DUCKDB_PATH, HISTORICAL_SEASONS


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH))


# ── Schema ─────────────────────────────────────────────────────────────────

CREATE_TABLES = """
-- Race schedule
CREATE TABLE IF NOT EXISTS race_schedule (
    season          INTEGER,
    round           INTEGER,
    race_name       VARCHAR,
    circuit_id      VARCHAR,
    circuit_name    VARCHAR,
    country         VARCHAR,
    date            DATE,
    PRIMARY KEY (season, round)
);

-- Race results
CREATE TABLE IF NOT EXISTS race_results (
    season            INTEGER,
    round             INTEGER,
    circuit_id        VARCHAR,
    driver_id         VARCHAR,
    driver_code       VARCHAR,
    constructor_id    VARCHAR,
    grid_position     INTEGER,
    finish_position   INTEGER,
    points_scored     DOUBLE,
    laps_completed    INTEGER,
    status            VARCHAR,
    is_classified     BOOLEAN,
    fastest_lap_rank  INTEGER,
    is_winner         BOOLEAN,
    is_top3           BOOLEAN,
    PRIMARY KEY (season, round, driver_id)
);

-- Qualifying
CREATE TABLE IF NOT EXISTS qualifying (
    season          INTEGER,
    round           INTEGER,
    circuit_id      VARCHAR,
    driver_id       VARCHAR,
    constructor_id  VARCHAR,
    quali_position  INTEGER,
    q1_time_s       DOUBLE,
    q2_time_s       DOUBLE,
    q3_time_s       DOUBLE,
    best_quali_s    DOUBLE,
    PRIMARY KEY (season, round, driver_id)
);

-- Driver standings
CREATE TABLE IF NOT EXISTS driver_standings (
    season    INTEGER,
    round     INTEGER,
    driver_id VARCHAR,
    position  INTEGER,
    points    DOUBLE,
    wins      INTEGER,
    PRIMARY KEY (season, round, driver_id)
);

-- Constructor standings
CREATE TABLE IF NOT EXISTS constructor_standings (
    season          INTEGER,
    round           INTEGER,
    constructor_id  VARCHAR,
    position        INTEGER,
    points          DOUBLE,
    wins            INTEGER,
    PRIMARY KEY (season, round, constructor_id)
);

-- FastF1 weather
CREATE TABLE IF NOT EXISTS race_weather (
    season        INTEGER,
    round         INTEGER,
    session_type  VARCHAR,
    air_temp_c    DOUBLE,
    track_temp_c  DOUBLE,
    humidity      DOUBLE,
    wind_speed    DOUBLE,
    is_wet_race   INTEGER,
    PRIMARY KEY (season, round, session_type)
);

-- FastF1 lap-1 sector-1 gaps
CREATE TABLE IF NOT EXISTS lap1_s1_gaps (
    season                INTEGER,
    round                 INTEGER,
    driver_code           VARCHAR,
    lap1_sector1_gap_s    DOUBLE,
    PRIMARY KEY (season, round, driver_code)
);
"""

CREATE_VIEWS = """
-- Master modelling view: one row per driver per race
CREATE OR REPLACE VIEW v_race_features AS
SELECT
    r.season,
    r.round,
    r.circuit_id,
    r.driver_id,
    r.driver_code,
    r.constructor_id,
    r.grid_position,
    r.finish_position,
    r.points_scored,
    r.is_winner,
    r.is_top3,
    r.status,
    -- Qualifying
    q.quali_position,
    q.best_quali_s,
    -- Gap to pole (quali)
    q.best_quali_s - MIN(q.best_quali_s) OVER (PARTITION BY r.season, r.round)
        AS quali_gap_to_pole_s,
    -- Championship standing BEFORE this race (round - 1)
    ds_prev.position    AS championship_position_pre,
    ds_prev.points      AS championship_points_pre,
    -- Constructor standing BEFORE this race
    cs_prev.position    AS constructor_position_pre,
    cs_prev.points      AS constructor_points_pre,
    -- Weather
    w.air_temp_c,
    w.track_temp_c,
    w.is_wet_race,
    -- Lap-1 sector-1 gap (race-week feature)
    s1.lap1_sector1_gap_s
FROM race_results r
LEFT JOIN qualifying q
    ON r.season = q.season AND r.round = q.round AND r.driver_id = q.driver_id
LEFT JOIN driver_standings ds_prev
    ON r.season = ds_prev.season
   AND r.round - 1 = ds_prev.round
   AND r.driver_id = ds_prev.driver_id
LEFT JOIN constructor_standings cs_prev
    ON r.season = cs_prev.season
   AND r.round - 1 = cs_prev.round
   AND r.constructor_id = cs_prev.constructor_id
LEFT JOIN race_weather w
    ON r.season = w.season AND r.round = w.round AND w.session_type = 'R'
LEFT JOIN lap1_s1_gaps s1
    ON r.season = s1.season AND r.round = s1.round AND r.driver_code = s1.driver_code;
"""


# ── Loader ─────────────────────────────────────────────────────────────────

def _load_parquet_if_exists(con: duckdb.DuckDBPyConnection,
                             path: Path, table: str) -> int:
    if not path.exists():
        return 0
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM read_parquet('{path}')")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return n
    except Exception as exc:
        logger.warning(f"  Failed loading {path.name} → {table}: {exc}")
        return 0


def build_warehouse(seasons: list[int] | None = None) -> None:
    seasons = seasons or HISTORICAL_SEASONS
    con = connect()

    logger.info("Creating tables …")
    con.executemany("", [])  # no-op; execute block below
    for stmt in CREATE_TABLES.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)

    for season in seasons:
        base = RAW_DIR / str(season)
        logger.info(f"Loading {season} …")

        tables = {
            "race_schedule":         base / "schedule.parquet",
            "race_results":          base / "results.parquet",
            "qualifying":            base / "qualifying.parquet",
            "driver_standings":      base / "driver_standings.parquet",
            "constructor_standings": base / "constructor_standings.parquet",
        }
        for table, path in tables.items():
            n = _load_parquet_if_exists(con, path, table)
            if n:
                logger.info(f"  {table}: {n} rows")

        # FastF1 sub-directory
        ff1 = base / "fastf1"
        _load_parquet_if_exists(con, ff1 / "weather.parquet",    "race_weather")
        _load_parquet_if_exists(con, ff1 / "lap1_s1_gaps.parquet", "lap1_s1_gaps")

    logger.info("Creating views …")
    for stmt in CREATE_VIEWS.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)

    logger.success(f"Warehouse built → {DUCKDB_PATH}")
    con.close()


def query(sql: str) -> pd.DataFrame:
    con = connect()
    df = con.execute(sql).df()
    con.close()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.build:
        build_warehouse()
    if args.query:
        df = query(args.query)
        print(df.to_string())
