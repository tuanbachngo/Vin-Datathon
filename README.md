# VIN Datathon

This repository contains the forecasting workflow, experiment notes, and submission artifacts used for the VIN Datathon sales-revenue task.

## Environment

Create or refresh the local virtual environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activate it in PowerShell with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, use the environment directly:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Launch JupyterLab with:

```powershell
jupyter lab
```

## Project layout

- `data/`: competition input files such as `sales.csv` and `sample_submission.csv`
- `notebooks/`: exploratory analysis and notebook-based modelling work
- `src/`: forecasting pipeline, feature engineering, validation, calibration, and model code
- `outputs/`: cross-validation reports, calibration artifacts, and intermediate experiment outputs
- `submissions/`: generated submission files for upload or comparison

## Forecast Pipeline

The primary run in this repository is the final pre-Covid anchor XGBoost pipeline below:

```powershell
.\.venv\Scripts\python.exe .\src\run_pipeline.py `
  --final-precovid-anchor `
  --precovid-feature-set anchor_gap `
  --regime-profile aggressive_w20_05 `
  --baseline-mode default `
  --submission-scale 1.15 `
  --submission-tag low_w2020_2022
```

This run trains the `xgboost` pre-Covid anchor model with:

- `anchor_gap` pre-Covid anchor features
- `aggressive_w20_05` regime weighting
- the default residual anchor baseline
- isotonic post-hoc calibration on out-of-fold predictions
- a final global submission scaling factor of `1.15`
- submission artifacts tagged as `low_w2020_2022`

Key artifacts for this run:

- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_oof_predictions.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_oof_calibrated.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_cv_results.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_cv_weighted_summary.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_calibration_summary.json`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_peak_month_error_summary.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_low_w2020_2022_run_meta.json`
- `submissions/submission_xgboost_precovid_anchor_calibrated_low_w2020_2022.csv`
- `submissions/submission_xgboost_precovid_anchor_scaled_low_w2020_2022.csv`

Current primary submission for upload:

- `submissions/submission_xgboost_precovid_anchor_scaled_low_w2020_2022.csv`
