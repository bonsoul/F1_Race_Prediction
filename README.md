# F1 2026 Race Prediction Platform

A production-style data science portfolio project: multi-source ingestion,
DuckDB feature store, three ML models, and a Streamlit app.

---

## Project structure

```
f1-predict/
├── config.py                   # Central config (paths, seasons, features, params)
├── pipeline.py                 # End-to-end orchestrator
├── requirements.txt
│
├── ingestion/
│   ├── ergast_loader.py        # Ergast API → Parquet (historical 2021–2025)
│   ├── fastf1_extractor.py     # FastF1 session extractor (lap times, weather)
│   └── openf1_poller.py        # OpenF1 live poller (race-week telemetry)
│
├── utils/
│   └── warehouse.py            # DuckDB warehouse + SQL views
│
├── features/
│   └── engineer.py             # ELO, rolling windows, circuit features
│
├── models/
│   ├── train.py                # Trains all three models
│   ├── predict.py              # Inference + SHAP explainability
│   └── saved/                  # Model pickle files (git-ignored)
│
├── app/
│   └── streamlit_app.py        # 4-tab Streamlit dashboard
│
├── data/
│   ├── raw/                    # Parquet files per season
│   ├── features/               # model_features.parquet
│   ├── fastf1_cache/           # FastF1 local cache
│   └── f1_platform.duckdb      # DuckDB warehouse
│
└── notebooks/                  # EDA notebooks (add your own)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Ingest 5 seasons + build warehouse + features + train + smoke-test
python pipeline.py --all --seasons 2021 2022 2023 2024 2025
```

Or step by step:

```bash
python pipeline.py --ingest    --seasons 2021 2022 2023 2024 2025
python pipeline.py --warehouse
python pipeline.py --features
python pipeline.py --train
python pipeline.py --predict
```

### 3. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Race-week live update

Run this on a race weekend to pull latest OpenF1 data:

```bash
# One-off pull
python pipeline.py --race-week

# Or schedule a continuous poll every 60 seconds during a live race
python -m ingestion.openf1_poller --mode schedule --interval 60
```

---

## Data sources

| Source | What it provides | Cadence |
|---|---|---|
| [Ergast API](https://ergast.com/mrd/) | Results, quali, standings, lap times (2021–2025) | Historical batch |
| [FastF1](https://docs.fastf1.dev/) | Structured session timing, telemetry, weather | Per-session |
| [OpenF1 API](https://openf1.org/) | Live telemetry, intervals, pit stops, weather | Race-week real-time |

---

## Models

| Model | Task | Algorithm | Key metric |
|---|---|---|---|
| Top-3 classifier | P(driver finishes in top 3) | XGBoost + Platt calibration | AUC, Brier score |
| Winner ranker | Rank all drivers by win probability | LightGBM LambdaRank | NDCG, Winner @1 |
| Points regressor | Expected points scored | XGBoost regression | MAE |

**Temporal split**: train on 2021–2024, validate on 2025. Never shuffle across seasons.

---

## Key features

| Feature | Source | Why it matters |
|---|---|---|
| `grid_position` | Ergast | Strongest single predictor |
| `quali_gap_to_pole_s` | Ergast + computed | Relative pace vs the field |
| `driver_rolling_elo` | Computed | Driver skill relative to opponents |
| `driver_rolling_points_5r` | Ergast + computed | Recent form |
| `track_overtaking_index` | Config | Adjusts grid→result mapping by circuit |
| `is_wet_race` | FastF1 weather | Wet races reshuffle the field |
| `constructor_pace_rank` | Computed | Team car performance this season |
| `lap1_sector1_gap_s` | FastF1/OpenF1 | Race-week predictor of true race pace |

---

## Extending the project

- Add **Sprint race** results as a separate target
- Add **tyre strategy** features (compound, age at each stint) from OpenF1 pit data
- Add **safety car probability** from historical race control data
- Add **driver head-to-head** teammate comparison features
- Deploy the Streamlit app to **Streamlit Cloud** (free tier works)
- Add **model monitoring**: track prediction calibration across 2026 races as results come in

---

## Notes on Ergast API

Ergast is a free, rate-limited API. The loader includes:
- 0.3s sleep between rounds to avoid hitting rate limits
- Exponential back-off retry on HTTP errors
- `limit=1000` parameter to pull all results per endpoint in one call

Ergast has announced end-of-life but remains operational at time of writing.
The OpenF1 + FastF1 combination can serve as a full replacement if needed.
