"""
models/predict.py
Inference layer — loads saved models and generates predictions
for a given race weekend.

Usage:
    from models.predict import RacePredictor
    predictor = RacePredictor()
    df = predictor.predict(race_features_df)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class RacePredictor:
    """
    Loads all three trained models and produces unified race predictions.

    Predictions per driver:
      - top3_probability    : P(finish in top 3)
      - winner_score        : LambdaRank score (higher = more likely to win)
      - predicted_points    : expected points scored
      - predicted_rank      : final ordering by winner_score
    """

    def __init__(self):
        self.top3_clf   = self._load("top3_classifier.pkl")
        self.ranker     = self._load("winner_ranker.pkl")
        self.pts_reg    = self._load("points_regressor.pkl")
        self._explainer = None   # lazy-init SHAP explainer

    @staticmethod
    def _load(filename: str):
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run `python -m models.train` first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        features : DataFrame with columns matching FEATURE_COLS,
                   plus id columns: driver_id, driver_code, constructor_id.

        Returns
        -------
        DataFrame sorted by predicted rank (winner first).
        """
        X = features[FEATURE_COLS].copy()

        # Impute any NaN with zeros (race-week features may be missing)
        X = X.fillna(X.median())

        # Top-3 probabilities
        top3_proba = self.top3_clf.predict_proba(X)[:, 1]

        # Winner ranking scores
        winner_score = self.ranker.predict(X.values)

        # Expected points
        pred_points = self.pts_reg.predict(X).clip(min=0)

        out = features[["driver_id", "driver_code", "constructor_id",
                          "grid_position", "quali_position"]].copy()
        out["top3_probability"]  = top3_proba.round(4)
        out["winner_score"]      = winner_score.round(4)
        out["predicted_points"]  = pred_points.round(2)
        out["predicted_rank"]    = out["winner_score"].rank(
            method="min", ascending=False
        ).astype(int)

        return out.sort_values("predicted_rank").reset_index(drop=True)

    def explain(self, features: pd.DataFrame,
                 n_drivers: int = 3) -> pd.DataFrame:
        """
        SHAP values for the Top-3 classifier on the top-n drivers.
        Returns a tidy DataFrame: (driver_code, feature, shap_value).
        """
        if self._explainer is None:
            # Unwrap calibrated classifier to get the underlying XGB model
            inner = self.top3_clf.calibrated_classifiers_[0].estimator
            self._explainer = shap.TreeExplainer(inner)

        X = features[FEATURE_COLS].fillna(0)
        shap_vals = self._explainer.shap_values(X)

        rows = []
        top_idx = (
            self.predict(features)
            .head(n_drivers)
            .merge(features.reset_index(), on=["driver_id", "driver_code"],
                   how="left")["index"]
            .tolist()
        )
        for idx in top_idx:
            driver = features.iloc[idx]["driver_code"]
            for feat, val in zip(FEATURE_COLS, shap_vals[idx]):
                rows.append({
                    "driver_code": driver,
                    "feature":     feat,
                    "shap_value":  round(float(val), 5),
                })
        return pd.DataFrame(rows)

    def predict_with_uncertainty(self, features: pd.DataFrame,
                                  n_bootstrap: int = 50) -> pd.DataFrame:
        """
        Bootstrap prediction intervals for top3_probability.
        Uses the individual calibrated classifiers in the ensemble.
        """
        X = features[FEATURE_COLS].fillna(0)
        clfs = self.top3_clf.calibrated_classifiers_

        all_probas = np.stack(
            [c.predict_proba(X)[:, 1] for c in clfs], axis=1
        )
        base = self.predict(features)
        base["top3_proba_mean"]   = all_probas.mean(axis=1).round(4)
        base["top3_proba_lower"]  = np.percentile(all_probas, 5,  axis=1).round(4)
        base["top3_proba_upper"]  = np.percentile(all_probas, 95, axis=1).round(4)
        return base
