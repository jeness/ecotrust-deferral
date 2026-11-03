# EcoTrust Dataset Schema

## Ground Truth
- `thermal_stress`: 1 if water_temp_c > 18°C (salmon thermal stress threshold)
- `high_migration_risk`: 1 if thermal_stress=1 AND month in Aug–Nov
- `water_temp_c`: **EXCLUDED from all agent input features** — target only

## Weather Agent Features (NOAA)
TMAX, TMIN, PRCP, AWND, SNOW, month, day_of_year

## Hydrology Agent Features (USGS — NO water_temp_c)
discharge_cfs, gage_height_ft, discharge_7d_mean, discharge_anomaly, month, day_of_year

## Governance Inputs (fused)
weather_risk, weather_ensemble_std, flood_risk, flood_ensemble_std,
confidence, entropy, conformal quantities, novelty score

No synthetic population or fish-count feature is used by the predictor or
governance layer.

## Dataset Statistics
- Date range:  1996-06-26 to 2024-12-31
- Total rows:  8,481
- Thermal stress days: 2,534 (29.9%)
- High migration risk days: 1,692 (20.0%)
- Migration season rows: 2,782
- Missing water_temp_c: 0 days
