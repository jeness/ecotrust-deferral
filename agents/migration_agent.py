"""
Migration Risk Agent — EcoTrust Pipeline

Task: predict HIGH MIGRATION RISK days during salmon migration season (Aug-Nov)
Ground truth: water_temp_c > 18°C AND month in [8,9,10,11]
              — measured by USGS, completely independent of input features

Features (no water_temp_c):
    weather_risk, weather_ensemble_std   (from WeatherAgent)
    flood_risk,   flood_ensemble_std     (from HydrologyAgent)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from uncertainty.ensemble import EnsembleClassifier
from uncertainty.conformal import SplitConformalClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = [
    "weather_risk", "weather_ensemble_std",
    "flood_risk",   "flood_ensemble_std",
]
CONFLICT_THRESHOLD = 0.35


class MigrationRiskAgent:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.ensemble = EnsembleClassifier()
        _cp_base = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=42, max_iter=1000)),
        ])
        self.cp = SplitConformalClassifier(_cp_base, alpha=alpha)
        self._ready = False

    def _merge(self, weather_df, hydro_df, merged_df=None):
        df = weather_df.rename(columns={"ensemble_std": "weather_ensemble_std"}).merge(
            hydro_df.rename(columns={"ensemble_std": "flood_ensemble_std"}),
            on="date", how="inner"
        )
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month

        if merged_df is not None:
            # Pull ground truth from the merged dataset.
            ref = merged_df[["date", "high_migration_risk"]].copy()
            ref["date"] = pd.to_datetime(ref["date"])
            df = df.merge(ref, on="date", how="left")
        else:
            df["high_migration_risk"] = np.nan

        df["conflict"] = (
            np.abs(df["weather_risk"] - df["flood_risk"]) > CONFLICT_THRESHOLD
        )
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
        return df

    def fit(self, weather_df, hydro_df, merged_df,
            cal_frac: float = 0.2) -> "MigrationRiskAgent":
        df = self._merge(weather_df, hydro_df, merged_df)
        # Train on ALL rows — month feature encodes seasonality
        # high_migration_risk = 0 outside Aug-Nov, model learns this pattern
        df = df[df["high_migration_risk"].notna()].copy()

        X = df[FEATURE_COLS].values
        y = df["high_migration_risk"].values.astype(int)
        split = int(len(X) * (1 - cal_frac))
        self.ensemble.fit(X[:split], y[:split])
        self.cp.fit(X[:split], y[:split], X[split:], y[split:])
        self._ready = True
        return self

    def predict(self, weather_df, hydro_df, merged_df=None) -> pd.DataFrame:
        if not self._ready:
            raise RuntimeError("Call fit() first.")
        df = self._merge(weather_df, hydro_df, merged_df)
        X = df[FEATURE_COLS].values
        ens = self.ensemble.predict(X)
        cp  = self.cp.predict(X)
        out = df[["date"]].copy()
        out["migration_risk"]  = ens["mean_prob"]
        out["ensemble_std"]    = ens["std_prob"]
        out["entropy"]         = ens["entropy"]
        out["cp_in_high_risk"] = cp["include_high_risk"]
        out["conflict"]        = df["conflict"].values
        out["ground_truth"]    = df["high_migration_risk"].values
        return out
