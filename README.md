# VIN Datathon

This workspace is set up for notebook-first data exploration and sales forecasting.

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
- `outputs/`: CV reports, calibration artifacts, and intermediate experiment outputs
- `submissions/`: generated submission files for upload or comparison

## Forecast Pipeline

Run the final pre-Covid anchor XGBoost pipeline from the repository root with:

```powershell
python src/run_pipeline.py --final-precovid-anchor --precovid-feature-set anchor_gap --regime-profile aggressive_w20_05 --baseline-mode default --submission-scale 1.05 --submission-tag anchor_gap_default_aggressive_w20_05_x105
```

This configuration builds the final `xgboost` pre-Covid anchor model with:

- `anchor_gap` pre-Covid anchor features
- `aggressive_w20_05` regime weighting
- the default residual anchor baseline
- isotonic post-hoc calibration on out-of-fold predictions
- a final global submission scaling factor of `1.05`

Key artifacts for this run:

- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_oof_predictions.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_oof_calibrated.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_cv_results.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_cv_weighted_summary.csv`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_calibration_summary.json`
- `outputs/xgboost_precovid_anchor_anchor_gap_aggressive_w20_05_default_anchor_gap_default_aggressive_w20_05_x105_run_meta.json`
- `submissions/submission_xgboost_precovid_anchor_calibrated_anchor_gap_default_aggressive_w20_05_x105.csv`
- `submissions/submission_xgboost_precovid_anchor_scaled_anchor_gap_default_aggressive_w20_05_x105.csv`

Best submission currently tracked in this repository:

- `submissions/submission_xgboost_precovid_anchor_scaled_anchor_gap_default_aggressive_w20_05_x105.csv`
