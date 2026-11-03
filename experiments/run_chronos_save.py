"""
Standalone Chronos runner — saves results row-by-row, resumable.
Run with: nohup python3 run_chronos_save.py > /tmp/chronos_log.txt 2>&1 &
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
sys.path.insert(0, "..")
import pandas as pd, numpy as np, torch
from pathlib import Path

DATA_DIR    = Path("../data")
OUT_FILE    = Path("../results/chronos_predictions.csv")
CONTEXT_LEN = 64
BATCH_SIZE  = 64
N_SAMPLES   = 10
THERMAL     = 18.0
MIGRATION_MONTHS = {8, 9, 10, 11}

def load_data():
    merged = pd.read_csv(DATA_DIR/"merged/columbia_merged.csv")
    merged["date"] = pd.to_datetime(merged["date"])
    merged["weather_risk"] = (merged["TMAX"] > 28).astype(float)
    merged["weather_ensemble_std"] = 0.01
    p20 = merged.groupby("month")["discharge_cfs"].quantile(0.2)
    merged["monthly_p20"] = merged["month"].map(p20.to_dict())
    merged["flood_risk"] = (merged["discharge_cfs"] < merged["monthly_p20"]).astype(float)
    merged["flood_ensemble_std"] = 0.01
    return merged

def main():
    print("Loading data...", flush=True)
    merged = load_data()
    train = merged[merged["date"].dt.year < 2015]
    test  = merged[merged["date"].dt.year >= 2015]
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)

    # Resume: skip already-done rows
    done_dates = set()
    if OUT_FILE.exists():
        done = pd.read_csv(OUT_FILE)
        done_dates = set(pd.to_datetime(done["date"]).astype(str))
        print(f"Resuming — {len(done_dates)} rows already done", flush=True)

    # Build full timeline for context lookup
    full = pd.concat([train, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    full["date"] = pd.to_datetime(full["date"])
    test_rows = full[full["date"].dt.year >= 2015].copy()
    test_rows = test_rows[~test_rows["date"].astype(str).isin(done_dates)]
    print(f"Rows to predict: {len(test_rows)}", flush=True)

    if len(test_rows) == 0:
        print("All done already!", flush=True)
        return

    print("Loading Chronos...", flush=True)
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(
        "/tmp/chronos-small", device_map="cpu", dtype=torch.float32)
    print("Chronos loaded", flush=True)

    # Write header if new file
    write_header = not OUT_FILE.exists()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    test_indices = test_rows.index.tolist()
    total = len(test_indices)
    t_start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_idxs = test_indices[batch_start: batch_start + BATCH_SIZE]
        contexts = []
        for idx in batch_idxs:
            ctx_series = full.iloc[max(0, idx - CONTEXT_LEN): idx]["water_temp_c"]
            ctx_arr = ctx_series.fillna(13.0).values.astype(np.float32)
            if len(ctx_arr) < CONTEXT_LEN:
                ctx_arr = np.pad(ctx_arr, (CONTEXT_LEN - len(ctx_arr), 0), constant_values=13.0)
            contexts.append(torch.tensor(ctx_arr[-CONTEXT_LEN:]))

        batch_tensor = torch.stack(contexts)
        forecast = pipe.predict(batch_tensor, prediction_length=1,
                                num_samples=N_SAMPLES, temperature=1.0,
                                limit_prediction_length=False)
        samples = forecast.squeeze(-1).numpy()   # (B, N_SAMPLES)
        risk_s  = (samples > THERMAL).astype(float)

        rows = []
        for i, idx in enumerate(batch_idxs):
            row = full.iloc[idx]
            in_season = row["month"] in MIGRATION_MONTHS
            mean_r = risk_s[i].mean() * (1.0 if in_season else 0.3)
            std_r  = risk_s[i].std()
            ent    = float(-mean_r * np.log(mean_r+1e-8) - (1-mean_r)*np.log(1-mean_r+1e-8))
            rows.append({
                "date":            str(row["date"].date()),
                "migration_risk":  round(float(mean_r), 6),
                "ensemble_std":    round(float(std_r), 6),
                "entropy":         round(ent, 6),
                "ground_truth":    row.get("high_migration_risk", np.nan),
                "month":           int(row["month"]),
            })

        df_batch = pd.DataFrame(rows)
        df_batch.to_csv(OUT_FILE, mode="a", header=write_header, index=False)
        write_header = False

        done_now = batch_start + len(batch_idxs)
        elapsed  = time.time() - t_start
        eta      = elapsed / done_now * (total - done_now)
        print(f"  {done_now}/{total} ({done_now*100//total}%) "
              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    print(f"\nDone! Results saved to {OUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
