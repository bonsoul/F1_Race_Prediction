"""
models/train.py
Trains three models with strict temporal train/val split:
  1. Top-3 finish classifier   (XGBoost, calibrated)
  2. Race winner ranker        (LightGBM LambdaRank)
  3. Points regression         (XGBoost)

Outputs saved to models/saved/:
  - top3_classifier.pkl
  - winner_ranker.pkl
  - points_regressor.pkl
  - eval_metrics.json

Usage:
    python -m models.train
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss,
    log_loss, mean_absolute_error, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_DIR, MODELS_DIR,
    TRAIN_SEASONS, VALIDATION_SEASON,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_TOP3, TARGET_WINNER, TARGET_POINTS,
    XGB_TOP3_PARAMS, XGB_POINTS_PARAMS, LGBM_RANK_PARAMS,
)

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Data loading ───────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_DIR / "model_features.parquet")
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val   = df[df["season"] == VALIDATION_SEASON].copy()
    logger.info(f"Train: {len(train)} rows | Val: {len(val)} rows")
    return train, val


# ── Evaluation helpers ─────────────────────────────────────────────────────

def eval_classifier(y_true, y_pred_proba, y_pred_binary, name: str) -> dict:
    metrics = {
        "auc":         round(roc_auc_score(y_true, y_pred_proba), 4),
        "log_loss":    round(log_loss(y_true, y_pred_proba), 4),
        "brier":       round(brier_score_loss(y_true, y_pred_proba), 4),
        "accuracy":    round(accuracy_score(y_true, y_pred_binary), 4),
    }
    logger.info(f"[{name}] " + " | ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def eval_regression(y_true, y_pred, name: str) -> dict:
    metrics = {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(((y_true - y_pred) ** 2).mean())), 4),
    }
    logger.info(f"[{name}] " + " | ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def _top_k_accuracy(df: pd.DataFrame, pred_col: str,
                     target_col: str, k: int = 3) -> float:
    """Per-race: did the top-k predicted drivers contain the actual winner?"""
    hits = []
    for _, grp in df.groupby(["season", "round"]):
        top_k_pred = set(grp.nlargest(k, pred_col)["driver_id"])
        actual = set(grp[grp[target_col] == 1]["driver_id"])
        hits.append(len(top_k_pred & actual) > 0)
    return round(float(np.mean(hits)), 4)


# ── Model 1: Top-3 classifier ──────────────────────────────────────────────

def train_top3(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    logger.info("── Training Top-3 classifier ──")
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_TOP3]
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_TOP3]

    base_clf = xgb.XGBClassifier(**XGB_TOP3_PARAMS)
    # Calibrate to get proper probabilities (Platt scaling)
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    clf.fit(X_train, y_train)

    proba   = clf.predict_proba(X_val)[:, 1]
    pred    = (proba >= 0.5).astype(int)
    metrics = eval_classifier(y_val, proba, pred, "Top-3 Classifier")

    # Top-3 hit rate: did our top-3 predicted drivers include the actual podium?
    val_copy = val.copy()
    val_copy["top3_proba"] = proba
    hit_rate = _top_k_accuracy(val_copy, "top3_proba", TARGET_TOP3, k=3)
    metrics["podium_hit_rate_@3"] = hit_rate
    logger.info(f"  Podium hit rate @3: {hit_rate}")

    # Save
    path = MODELS_DIR / "top3_classifier.pkl"
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    logger.success(f"  Saved → {path}")

    return metrics


# ── Model 2: Winner ranker (LightGBM LambdaRank) ──────────────────────────

def train_winner_ranker(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    logger.info("── Training Winner ranker (LambdaRank) ──")

    # LambdaRank needs group sizes (number of drivers per race)
    def _groups(df: pd.DataFrame) -> list[int]:
        return df.groupby(["season", "round"]).size().tolist()

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_WINNER].values.astype(int)
    g_train = _groups(train)

    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_WINNER].values.astype(int)
    g_val   = _groups(val)

    ranker = lgb.LGBMRanker(**LGBM_RANK_PARAMS)
    ranker.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    scores = ranker.predict(X_val)
    val_copy = val.copy()
    val_copy["winner_score"] = scores

    # Winner top-1 accuracy
    win1 = _top_k_accuracy(val_copy, "winner_score", TARGET_WINNER, k=1)
    win3 = _top_k_accuracy(val_copy, "winner_score", TARGET_WINNER, k=3)
    metrics = {"winner_acc_@1": win1, "winner_acc_@3": win3}
    logger.info(f"  Winner @1={win1} | Winner @3={win3}")

    path = MODELS_DIR / "winner_ranker.pkl"
    with open(path, "wb") as f:
        pickle.dump(ranker, f)
    logger.success(f"  Saved → {path}")

    return metrics


# ── Model 3: Points regressor ─────────────────────────────────────────────

def train_points_regressor(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    logger.info("── Training Points regressor ──")
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_POINTS]
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_POINTS]

    reg = xgb.XGBRegressor(**XGB_POINTS_PARAMS)
    reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    pred    = reg.predict(X_val).clip(min=0)
    metrics = eval_regression(y_val, pred, "Points Regressor")

    path = MODELS_DIR / "points_regressor.pkl"
    with open(path, "wb") as f:
        pickle.dump(reg, f)
    logger.success(f"  Saved → {path}")

    return metrics


# ── Feature importance ─────────────────────────────────────────────────────

def save_feature_importance(model_path: Path, feature_cols: list[str]) -> None:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Unwrap calibrated classifier
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
    else:
        inner = model

    if hasattr(inner, "feature_importances_"):
        imp = pd.DataFrame({
            "feature":    feature_cols,
            "importance": inner.feature_importances_,
        }).sort_values("importance", ascending=False)
        out_path = model_path.with_suffix(".feature_importance.csv")
        imp.to_csv(out_path, index=False)
        logger.info(f"  Feature importance → {out_path}")
        logger.info(f"\n{imp.head(10).to_string(index=False)}")


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_splits()

    all_metrics = {}
    all_metrics["top3_classifier"]   = train_top3(train, val)
    all_metrics["winner_ranker"]      = train_winner_ranker(train, val)
    all_metrics["points_regressor"]   = train_points_regressor(train, val)

    # Save combined metrics
    metrics_path = MODELS_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.success(f"Eval metrics → {metrics_path}")

    # Feature importance for tree models
    save_feature_importance(
        MODELS_DIR / "top3_classifier.pkl", FEATURE_COLS
    )
    save_feature_importance(
        MODELS_DIR / "points_regressor.pkl", FEATURE_COLS
    )

    return all_metrics


if __name__ == "__main__":
    run()
