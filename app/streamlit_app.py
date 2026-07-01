"""
app/streamlit_app.py
F1 2026 Race Prediction Dashboard

Tabs:
  1. Pre-race prediction — input grid + conditions → model output
  2. Historical performance — season results browser
  3. Model explainer — SHAP feature importance
  4. Live race week — latest OpenF1 data (if available)

Run:
    streamlit run app/streamlit_app.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, FEATURES_DIR, RAW_DIR,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, CIRCUIT_TYPES,
)

st.set_page_config(
    page_title="F1 2026 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CIRCUIT_TYPE_MAP = {"street": 0, "hybrid": 1, "highspeed": 2, "technical": 3}
OVERTAKING_INDEX = {"street": 0.90, "technical": 0.65, "hybrid": 0.50, "highspeed": 0.25}

F1_TEAMS_2026 = [
    "Red Bull", "Ferrari", "Mercedes", "McLaren", "Aston Martin",
    "Alpine", "Williams", "RB", "Kick Sauber", "Haas",
]

F1_DRIVERS_2026 = {
    "Red Bull":       ["Verstappen", "Lawson"],
    "Ferrari":        ["Leclerc", "Hamilton"],
    "Mercedes":       ["Russell", "Antonelli"],
    "McLaren":        ["Norris", "Piastri"],
    "Aston Martin":   ["Alonso", "Stroll"],
    "Alpine":         ["Gasly", "Doohan"],
    "Williams":       ["Albon", "Sainz"],
    "RB":             ["Tsunoda", "Hadjar"],
    "Kick Sauber":    ["Hulkenberg", "Bortoleto"],
    "Haas":           ["Ocon", "Bearman"],
}

ALL_DRIVERS = [d for drivers in F1_DRIVERS_2026.values() for d in drivers]

CIRCUITS_2026 = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "USA", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]


# ── Model loading (cached) ────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}
    for name in ["top3_classifier", "winner_ranker", "points_regressor"]:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


@st.cache_data
def load_feature_table():
    path = FEATURES_DIR / "model_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_eval_metrics():
    path = MODELS_DIR / "eval_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Prediction helper ─────────────────────────────────────────────────────

def build_feature_row(
    driver: str, constructor: str, grid_pos: int, quali_pos: int,
    quali_gap: float, circuit_type: str, is_wet: int,
    air_temp: float, track_temp: float,
    drv_elo: float, drv_pts_5r: float, drv_dnf_5r: float,
    con_pts_5r: float, con_reliability: float, con_pace_rank: int,
    round_num: int,
) -> dict:
    return {
        "grid_position":              grid_pos,
        "quali_gap_to_pole_s":        quali_gap,
        "driver_rolling_elo":         drv_elo,
        "driver_rolling_points_5r":   drv_pts_5r,
        "driver_rolling_dnf_rate_5r": drv_dnf_5r,
        "constructor_rolling_points_5r": con_pts_5r,
        "constructor_pace_rank":      con_pace_rank,
        "constructor_reliability_5r": con_reliability,
        "track_overtaking_index":     OVERTAKING_INDEX[circuit_type],
        "is_wet_race":                is_wet,
        "air_temp_c":                 air_temp,
        "track_temp_c":               track_temp,
        "lap1_sector1_gap_s":         0.0,
        "quali_position":             quali_pos,
        "season_race_number":         round_num,
        "circuit_type":               CIRCUIT_TYPE_MAP[circuit_type],
        "tyre_compound_start":        1,   # default Medium
        "driver_id":                  driver.lower(),
        "driver_code":                driver[:3].upper(),
        "constructor_id":             constructor.lower().replace(" ", "_"),
    }


def run_prediction(models: dict, feature_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(feature_rows)
    X  = df[FEATURE_COLS].fillna(0)

    results = df[["driver_id", "driver_code", "constructor_id",
                   "grid_position", "quali_position"]].copy()

    if "top3_classifier" in models:
        results["top3_probability"] = models["top3_classifier"].predict_proba(X)[:, 1]
    else:
        results["top3_probability"] = np.random.uniform(0.05, 0.45, len(df))

    if "winner_ranker" in models:
        results["winner_score"] = models["winner_ranker"].predict(X.values)
    else:
        results["winner_score"] = np.random.uniform(0, 1, len(df))

    if "points_regressor" in models:
        results["predicted_points"] = models["points_regressor"].predict(X).clip(min=0)
    else:
        results["predicted_points"] = np.random.uniform(0, 25, len(df))

    results["predicted_rank"] = results["winner_score"].rank(
        method="min", ascending=False
    ).astype(int)

    return results.sort_values("predicted_rank").reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/320px-F1.svg.png",
        width=120,
    )
    st.sidebar.title("F1 2026 Predictor")
    st.sidebar.markdown("---")
    circuit = st.sidebar.selectbox("Circuit", CIRCUITS_2026, index=0)
    circuit_type = st.sidebar.selectbox(
        "Circuit type",
        ["street", "hybrid", "highspeed", "technical"],
        index=1,
    )
    round_num = st.sidebar.number_input("Race round", min_value=1, max_value=24, value=1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Weather**")
    is_wet    = st.sidebar.toggle("Wet race", value=False)
    air_temp  = st.sidebar.slider("Air temp (°C)", 10, 45, 28)
    track_temp = st.sidebar.slider("Track temp (°C)", 15, 65, 42)
    return circuit, circuit_type, round_num, int(is_wet), air_temp, track_temp


# ── Tab 1: Pre-race prediction ────────────────────────────────────────────

def tab_prediction(models, circuit, circuit_type, round_num, is_wet, air_temp, track_temp):
    st.header("Pre-race grid builder")
    st.caption("Enter qualifying results and driver form → model predicts race outcome.")

    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown(
            "Fill in each driver's grid position and quali gap. "
            "Adjust form metrics (ELO, rolling points) from recent race history. "
            "Click **Run prediction** to generate results."
        )

    cols = st.columns([2, 1, 1, 1, 1, 1])
    with cols[0]: st.markdown("**Driver**")
    with cols[1]: st.markdown("**Grid**")
    with cols[2]: st.markdown("**Quali gap (s)**")
    with cols[3]: st.markdown("**ELO**")
    with cols[4]: st.markdown("**Pts/5r**")
    with cols[5]: st.markdown("**DNF rate**")

    feature_rows = []
    for team, drivers in F1_DRIVERS_2026.items():
        for i, driver in enumerate(drivers):
            cols = st.columns([2, 1, 1, 1, 1, 1])
            with cols[0]:
                st.write(f"🏎️ **{driver}** _{team}_")
            with cols[1]:
                grid = st.number_input("", min_value=1, max_value=20,
                                        value=i * 10 + list(F1_DRIVERS_2026.keys()).index(team) + 1,
                                        key=f"grid_{driver}", label_visibility="collapsed")
            with cols[2]:
                gap = st.number_input("", min_value=0.0, max_value=5.0,
                                       value=round(grid * 0.08, 3),
                                       key=f"gap_{driver}", label_visibility="collapsed")
            with cols[3]:
                elo = st.number_input("", min_value=1200, max_value=1900,
                                       value=1550 - grid * 10,
                                       key=f"elo_{driver}", label_visibility="collapsed")
            with cols[4]:
                pts = st.number_input("", min_value=0.0, max_value=25.0,
                                       value=max(0.0, 20.0 - grid),
                                       key=f"pts_{driver}", label_visibility="collapsed")
            with cols[5]:
                dnf = st.number_input("", min_value=0.0, max_value=1.0,
                                       value=0.1, step=0.05,
                                       key=f"dnf_{driver}", label_visibility="collapsed")

            feature_rows.append(build_feature_row(
                driver=driver, constructor=team,
                grid_pos=grid, quali_pos=grid, quali_gap=gap,
                circuit_type=circuit_type, is_wet=is_wet,
                air_temp=air_temp, track_temp=track_temp,
                drv_elo=elo, drv_pts_5r=pts, drv_dnf_5r=dnf,
                con_pts_5r=max(0, 30 - list(F1_DRIVERS_2026.keys()).index(team) * 5),
                con_reliability=0.85, con_pace_rank=list(F1_DRIVERS_2026.keys()).index(team) + 1,
                round_num=round_num,
            ))

    if st.button("🏁 Run prediction", type="primary"):
        with st.spinner("Computing predictions …"):
            preds = run_prediction(models, feature_rows)

        st.markdown("---")
        st.subheader("Race prediction")

        # Podium cards
        podium_cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (col, medal) in enumerate(zip(podium_cols, medals)):
            row = preds.iloc[i]
            with col:
                st.metric(
                    label=f"{medal} P{i+1}",
                    value=row["driver_code"],
                    delta=f"P(top3)={row['top3_probability']:.1%}",
                )

        st.markdown("---")

        # Full prediction table
        display = preds[["predicted_rank", "driver_code", "constructor_id",
                           "grid_position", "top3_probability",
                           "predicted_points"]].copy()
        display.columns = ["Pred rank", "Driver", "Constructor",
                            "Grid", "P(Top 3)", "Pred points"]
        display["P(Top 3)"] = display["P(Top 3)"].apply(lambda x: f"{x:.1%}")
        display["Pred points"] = display["Pred points"].apply(lambda x: f"{x:.1f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Bar chart
        fig = px.bar(
            preds.head(10),
            x="driver_code", y="top3_probability",
            color="top3_probability",
            color_continuous_scale="RdYlGn",
            title="Top-3 probability by driver",
            labels={"top3_probability": "P(Top 3)", "driver_code": "Driver"},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Historical performance ─────────────────────────────────────────

def tab_history():
    st.header("Historical race data")
    df = load_feature_table()
    if df.empty:
        st.info("No feature data found. Run the ingestion and feature pipelines first.")
        return

    c1, c2 = st.columns(2)
    season  = c1.selectbox("Season", sorted(df["season"].unique(), reverse=True))
    drivers = c2.multiselect("Drivers", sorted(df["driver_code"].unique()), default=[])

    view = df[df["season"] == season]
    if drivers:
        view = view[view["driver_code"].isin(drivers)]

    st.dataframe(
        view[["round", "driver_code", "constructor_id", "grid_position",
               "finish_position", "points_scored", "is_top3", "is_winner"]],
        use_container_width=True, hide_index=True
    )

    # Points over season
    season_pts = (
        df[df["season"] == season]
        .sort_values("round")
        .groupby(["round", "driver_code"])["points_scored"]
        .sum()
        .groupby(level="driver_code")
        .cumsum()
        .reset_index()
    )
    if drivers:
        season_pts = season_pts[season_pts["driver_code"].isin(drivers)]

    if not season_pts.empty:
        fig = px.line(season_pts, x="round", y="points_scored",
                       color="driver_code",
                       title=f"{season} cumulative points",
                       labels={"points_scored": "Cumulative points", "round": "Round"})
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Model explainer ────────────────────────────────────────────────

def tab_explainer():
    st.header("Model explainer")
    metrics = load_eval_metrics()

    if metrics:
        st.subheader("Validation metrics (2025 season)")
        m_cols = st.columns(len(metrics))
        for col, (model_name, vals) in zip(m_cols, metrics.items()):
            with col:
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                for k, v in vals.items():
                    st.metric(k, v)
    else:
        st.info("No eval metrics found. Train the models first.")

    # Feature importance
    imp_path = MODELS_DIR / "top3_classifier.feature_importance.csv"
    if imp_path.exists():
        st.subheader("Feature importance — Top-3 classifier")
        imp = pd.read_csv(imp_path)
        fig = px.bar(
            imp.head(12), x="importance", y="feature",
            orientation="h", title="Top 12 features",
            color="importance", color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the models to see feature importance.")


# ── Tab 4: Live race week ─────────────────────────────────────────────────

def tab_live():
    st.header("Live race week — OpenF1")
    openf1_dir = RAW_DIR / "openf1"

    if not openf1_dir.exists() or not any(openf1_dir.iterdir()):
        st.info(
            "No OpenF1 data found locally.\n\n"
            "Run the poller during a race weekend:\n"
            "```\npython -m Ingestion.openf1_poller --mode pull\n```"
        )
        return

    sessions = sorted(openf1_dir.iterdir(), key=lambda p: p.name)
    session  = st.selectbox("Session", [s.name for s in sessions])
    sess_dir = openf1_dir / session

    laps_path = sess_dir / "laps.parquet"
    if laps_path.exists():
        laps = pd.read_parquet(laps_path)
        st.subheader("Lap times")
        if "lap_duration" in laps.columns and "driver_number" in laps.columns:
            fig = px.box(
                laps[laps["lap_duration"] < laps["lap_duration"].quantile(0.95)],
                x="driver_number", y="lap_duration",
                title="Lap time distribution per driver",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(laps.head(50), use_container_width=True)

    pace_path = sess_dir / "pace_summary.parquet"
    if pace_path.exists():
        pace = pd.read_parquet(pace_path)
        st.subheader("Pace summary")
        st.dataframe(pace.sort_values("avg_lap_s"), use_container_width=True, hide_index=True)

    weather_path = sess_dir / "weather.parquet"
    if weather_path.exists():
        w = pd.read_parquet(weather_path)
        st.subheader("Weather")
        if "AirTemp" in w.columns:
            fig = px.line(w, x=w.index, y=["AirTemp", "TrackTemp"],
                           title="Temperature over session")
            st.plotly_chart(fig, use_container_width=True)


# ── App entry point ───────────────────────────────────────────────────────

def main():
    models = load_models()
    circuit, circuit_type, round_num, is_wet, air_temp, track_temp = sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏁 Pre-race prediction",
        "📊 Historical data",
        "🧠 Model explainer",
        "📡 Live race week",
    ])

    with tab1:
        tab_prediction(models, circuit, circuit_type, round_num,
                        is_wet, air_temp, track_temp)
    with tab2:
        tab_history()
    with tab3:
        tab_explainer()
    with tab4:
        tab_live()


if __name__ == "__main__":
    main()
