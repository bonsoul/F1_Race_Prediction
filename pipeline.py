"""
pipeline.py
End-to-end pipeline runner for the F1 2026 prediction platform.

Steps:
  1. ingest    â€” pull historical data from Ergast + FastF1
  2. warehouse â€” load Parquet files into DuckDB
  3. features  â€” engineer model features
  4. train     â€” train all three models
  5. predict   â€” smoke-test predictions on latest season

Usage:
    # Run all steps
    python pipeline.py --all

    # Run individual steps
    python pipeline.py --ingest --seasons 2021 2022 2023 2024 2025
    python pipeline.py --warehouse
    python pipeline.py --features
    python pipeline.py --train
    python pipeline.py --predict

    # Full 2026 race-week update (ingest OpenF1 + predict)
    python pipeline.py --race-week
"""
import argparse
from loguru import logger


def step_ingest(seasons: list[int]) -> None:
    logger.info("═══ STEP 1: Ergast ingestion ═══")
    from Ingestion.ergast_loader import run_season
    ergast_failed: list[int] = []
    for s in seasons:
        try:
            ok = run_season(s)
            if ok is False:
                ergast_failed.append(s)
        except Exception:
            ergast_failed.append(s)
            logger.exception(f"[{s}] Ergast ingestion failed unexpectedly")

    if ergast_failed:
        logger.warning(f"Ergast ingestion skipped for seasons: {ergast_failed}")

    logger.info("═══ STEP 1b: FastF1 extraction ═══")
    from Ingestion.fastf1_extractor import run as ff1_run
    fastf1_failed: list[int] = []
    for s in seasons:
        try:
            ff1_run(s, list(range(1, 25)))
        except Exception:
            fastf1_failed.append(s)
            logger.exception(f"[{s}] FastF1 extraction failed unexpectedly")

    if fastf1_failed:
        logger.warning(f"FastF1 extraction failed for seasons: {fastf1_failed}")
def step_warehouse() -> None:
    logger.info("â•â•â• STEP 2: Build DuckDB warehouse â•â•â•")
    from utils.warehouse import build_warehouse
    build_warehouse()


def step_features() -> None:
    logger.info("â•â•â• STEP 3: Feature engineering â•â•â•")
    from features.engineer import build_features
    df = build_features()
    logger.success(f"Features ready: {df.shape}")


def step_train() -> None:
    logger.info("â•â•â• STEP 4: Model training â•â•â•")
    from models.train import run
    metrics = run()
    for model, m in metrics.items():
        logger.info(f"  {model}: {m}")


def step_predict() -> None:
    logger.info("â•â•â• STEP 5: Smoke-test prediction â•â•â•")
    import pandas as pd
    from models.predict import RacePredictor
    from config import FEATURES_DIR, VALIDATION_SEASON

    df = pd.read_parquet(FEATURES_DIR / "model_features.parquet")
    # Use round 1 of the validation season as a demo
    sample = df[(df["season"] == VALIDATION_SEASON) & (df["round"] == 1)].copy()

    if sample.empty:
        logger.warning(f"No data for {VALIDATION_SEASON} round 1 â€” skipping smoke test.")
        return

    predictor = RacePredictor()
    preds = predictor.predict(sample)
    logger.info(f"\n{preds[['driver_code','predicted_rank','top3_probability','predicted_points']].to_string(index=False)}")


def step_race_week() -> None:
    logger.info("â•â•â• RACE WEEK: OpenF1 live pull â•â•â•")
    from Ingestion.openf1_poller import pull_latest
    pull_latest("Race")
    pull_latest("Qualifying")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 2026 prediction pipeline")
    parser.add_argument("--all",       action="store_true", help="Run all steps")
    parser.add_argument("--ingest",    action="store_true")
    parser.add_argument("--warehouse", action="store_true")
    parser.add_argument("--features",  action="store_true")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--predict",   action="store_true")
    parser.add_argument("--race-week", action="store_true")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()

    if args.all:
        step_ingest(args.seasons)
        step_warehouse()
        step_features()
        step_train()
        step_predict()
    else:
        if args.ingest:    step_ingest(args.seasons)
        if args.warehouse: step_warehouse()
        if args.features:  step_features()
        if args.train:     step_train()
        if args.predict:   step_predict()
        if args.race_week: step_race_week()

    logger.success("Pipeline complete.")
