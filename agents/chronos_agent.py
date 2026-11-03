"""
Chronos Migration Agent — EcoTrust Phase 2 Backend

Replaces the RF/XGB Migration Risk Agent with Amazon Chronos,
a zero-shot time-series foundation model.

Strategy:
    1. Frame the task as time-series forecasting:
       Use past N days of (weather_risk, flood_risk, month) to
       forecast next-day water_temp_c.
    2. High forecasted water_temp (→ >18°C) = high migration risk.
    3. Chronos prediction intervals → uncertainty signal (replaces ensemble_std).
    4. The Decision Governance Layer (escalation policy) is UNCHANGED.

This demonstrates EcoTrust's model-agnostic design:
    swapping the prediction backend leaves the governance layer intact.

Install:
    pip install chronos-forecasting torch

Usage:
    agent = ChronosMigrationAgent()
    agent.fit(merged_train)
    out = agent.predict(merged_test)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from uncertainty.calibration import expected_calibration_error

CONTEXT_LEN = 64    # days of history Chronos sees
FORECAST_LEN = 1    # predict next day


class ChronosMigrationAgent:
    """
    Chronos-based migration risk agent.
    Predicts water temperature as a time-series forecasting task,
    then converts temperature forecast to risk probability.
    """

    THERMAL_THRESHOLD = 18.0   # °C — salmon thermal stress
    MIGRATION_MONTHS  = {8, 9, 10, 11}

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        """
        Args:
            model_size: "tiny" | "small" | "base" | "large"
                        "small" (≈46M params) runs on CPU in <2s/sample
            device:     "cpu" | "cuda"
        """
        self.model_size = model_size
        self.device     = device
        self.pipeline   = None
        self._ready     = False

    def _load_model(self):
        try:
            from chronos import ChronosPipeline
            import torch
            # Use local path if available, otherwise download
            local_path = f"/tmp/chronos-{self.model_size}"
            model_id = local_path if Path(local_path).exists() else f"amazon/chronos-t5-{self.model_size}"
            self.pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=torch.float32,
            )
            self._ready = True
            print(f"  Chronos ({self.model_size}) loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "Install chronos: pip install chronos-forecasting torch"
            )

    def fit(self, merged_df: pd.DataFrame) -> "ChronosMigrationAgent":
        """
        Chronos is zero-shot — no training required.
        We use the training set only to compute the water_temp_c
        normalization statistics needed for thresholding.
        """
        if not self._ready:
            self._load_model()
        merged_df = merged_df.copy()
        merged_df["date"] = pd.to_datetime(merged_df["date"])
        self._train_df = merged_df.sort_values("date")
        print("  Chronos: zero-shot, no training. Context window = last",
              CONTEXT_LEN, "days.")
        return self

    def _build_series(self, history_df: pd.DataFrame) -> np.ndarray:
        """Build the univariate water_temp_c series for Chronos context."""
        series = history_df["water_temp_c"].fillna(method="ffill").fillna(13.0).values
        return series.astype(np.float32)

    def predict(self, merged_test: pd.DataFrame) -> pd.DataFrame:
        """
        Batched prediction: build one context tensor per test row,
        stack into a batch, run Chronos once per batch of 32.
        ~10x faster than row-by-row on CPU.
        """
        if not self._ready:
            self._load_model()

        import torch

        full_df = pd.concat([self._train_df, merged_test], ignore_index=True)
        full_df = full_df.sort_values("date").reset_index(drop=True)
        full_df["date"] = pd.to_datetime(full_df["date"])

        test_idx = full_df[full_df["date"].isin(
            pd.to_datetime(merged_test["date"]))].index.tolist()

        N_SAMPLES  = 10   # reduced from 20 for speed
        BATCH_SIZE = 64   # process 64 rows at a time

        all_mean_risk = []
        all_std_risk  = []

        print(f"  Chronos predicting {len(test_idx)} test points "
              f"(batch={BATCH_SIZE}, samples={N_SAMPLES})...")

        for batch_start in range(0, len(test_idx), BATCH_SIZE):
            batch_idxs = test_idx[batch_start: batch_start + BATCH_SIZE]
            contexts = []
            for idx in batch_idxs:
                ctx = full_df.iloc[max(0, idx - CONTEXT_LEN): idx]
                series = self._build_series(ctx) if len(ctx) >= 2 else np.full(CONTEXT_LEN, 13.0, dtype=np.float32)
                # Pad/truncate to fixed length for batching
                if len(series) < CONTEXT_LEN:
                    series = np.pad(series, (CONTEXT_LEN - len(series), 0), constant_values=13.0)
                contexts.append(torch.tensor(series[-CONTEXT_LEN:]))

            batch_tensor = torch.stack(contexts)  # (B, CONTEXT_LEN)
            forecast = self.pipeline.predict(
                batch_tensor,
                prediction_length=FORECAST_LEN,
                num_samples=N_SAMPLES,
                temperature=1.0,
                limit_prediction_length=False,
            )  # shape: (B, N_SAMPLES, 1)
            samples = forecast.squeeze(-1).numpy()  # (B, N_SAMPLES)
            risk_samples = (samples > self.THERMAL_THRESHOLD).astype(float)
            all_mean_risk.extend(risk_samples.mean(axis=1).tolist())
            all_std_risk.extend(risk_samples.std(axis=1).tolist())

            done = min(batch_start + BATCH_SIZE, len(test_idx))
            print(f"    {done}/{len(test_idx)} ({done*100//len(test_idx)}%)", flush=True)

        results = []
        for i, idx in enumerate(test_idx):
            row = full_df.iloc[idx]
            migration_season = row["month"] in self.MIGRATION_MONTHS
            mean_risk = all_mean_risk[i] * (1.0 if migration_season else 0.3)
            std_risk  = all_std_risk[i]
            entropy   = float(-mean_risk * np.log(mean_risk + 1e-8)
                               - (1 - mean_risk) * np.log(1 - mean_risk + 1e-8))
            results.append({
                "date":            row["date"],
                "migration_risk":  float(mean_risk),
                "ensemble_std":    float(std_risk),
                "entropy":         entropy,
                "cp_in_high_risk": bool(mean_risk > 0.5),
                "conflict":        False,
                "ground_truth":    row.get("high_migration_risk", np.nan),
            })

        return pd.DataFrame(results)


def run_chronos_comparison(
    merged_train: pd.DataFrame,
    merged_test:  pd.DataFrame,
    governance_config=None,
):
    """
    Run Chronos backend through EcoTrust governance layer.
    Returns metrics dict for inclusion in the Phase 2 comparison table.
    """
    from governance.escalation import HumanGovernanceLayer, EscalationConfig
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    agent = ChronosMigrationAgent(model_size="small", device="cpu")
    agent.fit(merged_train)
    out = agent.predict(merged_test)

    gov = HumanGovernanceLayer(governance_config or EscalationConfig())
    gov_df  = gov.evaluate(out)
    gov_sum = gov.summary()

    valid = out.dropna(subset=["ground_truth"])
    y_true = valid["ground_truth"].values.astype(int)
    y_prob = valid["migration_risk"].values
    y_pred = (y_prob > 0.5).astype(int)

    cal = expected_calibration_error(y_true, y_prob)

    metrics = {
        "backend":          "Chronos-small",
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else None,
        "ece":              cal["ece"],
        "brier_score":      cal["brier_score"],
        "escalation_rate":  gov_sum.get("escalation_rate", 0),
        "error_coverage":   gov_sum.get("error_coverage", np.nan),
        "error_reduction":  gov_sum.get("error_reduction", np.nan),
        "accuracy_after":   gov_sum.get("accuracy_after", np.nan),
    }
    return metrics, out, gov_df
