"""
Weather Agent — EcoTrust Pipeline

Task: predict probability of thermally stressful air conditions
      (TMAX > 28°C as proxy for river heating risk)
Features: NOAA only — TMAX, TMIN, PRCP, AWND, SNOW, month, day_of_year
Target: air_heat_risk = (TMAX > 28) — independent of water_temp_c
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from uncertainty.ensemble import EnsembleRegressor
from uncertainty.conformal import SplitConformalRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLS = ["TMAX", "TMIN", "PRCP", "AWND", "SNOW"]  # no temporal features


class WeatherAgent:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.ensemble = EnsembleRegressor()
        _base = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        self.cp = SplitConformalRegressor(_base, alpha=alpha)
        self._ready = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.day_of_year
        # Target: air temp heating risk — proxy for stream heating
        df["air_heat_risk"] = (df["TMAX"] > 28.0).astype(float)
        for col in ["TMAX", "TMIN", "PRCP", "AWND", "SNOW"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                median = df[col].median()
                df[col] = df[col].fillna(0.0 if pd.isna(median) else median)
        return df

    def fit(self, df: pd.DataFrame, cal_frac: float = 0.2) -> "WeatherAgent":
        df = self._build_features(df)
        df = df.dropna(subset=["air_heat_risk"])
        X = df[FEATURE_COLS].values
        y = df["air_heat_risk"].values
        split = int(len(X) * (1 - cal_frac))
        self.ensemble.fit(X[:split], y[:split])
        self.cp.fit(X[:split], y[:split], X[split:], y[split:])
        self._ready = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._ready:
            raise RuntimeError("Call fit() first.")
        df = self._build_features(df)
        X = df[FEATURE_COLS].values
        ens = self.ensemble.predict(X)
        cp  = self.cp.predict(X)
        out = df[["date"]].copy()
        out["weather_risk"]   = ens["mean_pred"].clip(0, 1)
        out["ensemble_std"]   = ens["std_pred"]
        out["cp_lower"]       = cp["lower"].clip(0, 1)
        out["cp_upper"]       = cp["upper"].clip(0, 1)
        out["interval_width"] = cp["width"]
        return out

    def simulate_sensor_failure(self, df, drop_features=None, drop_frac=0.3):
        df = df.copy()
        if drop_features is None:
            drop_features = ["PRCP", "AWND"]
        mask = np.random.random(len(df)) < drop_frac
        for col in drop_features:
            if col in df.columns:
                df.loc[mask, col] = np.nan
        return df

    def add_noise(self, df, noise_frac=0.1):
        df = df.copy()
        for col in ["TMAX", "TMIN", "PRCP"]:
            if col in df.columns:
                sigma = noise_frac * df[col].std()
                df[col] = df[col] + np.random.normal(0, sigma, len(df))
        return df
