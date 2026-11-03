"""
Hydrology Agent — EcoTrust Pipeline

Task: predict low-flow stress probability
      (discharge < 20th percentile for that month)
Features: USGS streamflow only — NO water_temp_c (that is the ground truth target)
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

# water_temp_c is intentionally excluded — it is the migration agent's ground truth
FEATURE_COLS = [
    "discharge_cfs", "gage_height_ft",
    "discharge_7d_mean", "discharge_anomaly",
]  # no temporal features — discharge anomaly encodes seasonality implicitly


class HydrologyAgent:
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
        self._monthly_p20: dict = {}

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.day_of_year
        df = df.sort_values("date")
        df["discharge_7d_mean"] = df["discharge_cfs"].rolling(7, min_periods=1).mean()

        if self._monthly_p20:
            df["monthly_p20"] = df["month"].map(self._monthly_p20)
        else:
            p20 = df.groupby("month")["discharge_cfs"].quantile(0.2)
            self._monthly_p20 = p20.to_dict()
            df["monthly_p20"] = df["month"].map(self._monthly_p20)

        df["discharge_anomaly"] = df["discharge_cfs"] - df["monthly_p20"]
        # Target: low-flow stress (low flow → reduced cooling, thermal stress risk)
        df["low_flow_risk"] = (df["discharge_cfs"] < df["monthly_p20"]).astype(float)

        if "gage_height_ft" not in df.columns:
            df["gage_height_ft"] = 0.0
        else:
            df["gage_height_ft"] = df["gage_height_ft"].fillna(0.0)

        discharge_median = df["discharge_cfs"].median()
        df["discharge_cfs"] = df["discharge_cfs"].fillna(
            0.0 if pd.isna(discharge_median) else discharge_median
        )
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(0.0)
        return df

    def fit(self, df: pd.DataFrame, cal_frac: float = 0.2) -> "HydrologyAgent":
        df = self._build_features(df)
        df = df.dropna(subset=["low_flow_risk"])
        X = df[FEATURE_COLS].values
        y = df["low_flow_risk"].values
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
        out["flood_risk"]     = ens["mean_pred"].clip(0, 1)
        out["ensemble_std"]   = ens["std_pred"]
        out["cp_lower"]       = cp["lower"].clip(0, 1)
        out["cp_upper"]       = cp["upper"].clip(0, 1)
        out["interval_width"] = cp["width"]
        return out

    def simulate_sensor_failure(self, df, drop_frac=0.3):
        df = df.copy()
        mask = np.random.random(len(df)) < drop_frac
        df.loc[mask, "gage_height_ft"] = np.nan
        df.loc[mask, "discharge_7d_mean"] = np.nan
        return df
