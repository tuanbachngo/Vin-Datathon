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

- `data/`: raw data files such as the CSV drop from the competition
- `outputs/`: charts, exports, and intermediate artifacts
- `submissions/`: generated submission files
- `data.ipynb`: starter notebook for EDA and modeling
- `archive/sql_mcq/`: archived SQL MCQ materials restored from the original repository

## Forecast Pipeline

Run the end-to-end forecasting script with:

```powershell
.\.venv\Scripts\python.exe -m src.run_pipeline
```

This now includes:

- the existing baseline, hist-GBM, GBR, LightGBM, and MLP models
- an `xgboost` top-aux feature pipeline
- post-hoc calibration for the `xgboost` submission using out-of-fold predictions

To tune the `xgboost` model with Optuna before generating CV results and submission files:

```powershell
.\.venv\Scripts\python.exe -m src.run_pipeline --tune-xgboost --xgb-trials 40
```

To run the `xgboost residual no lag` branch with the legacy seasonal-structure
lookup and MIC-ranked auxiliary feature selection:

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py `
  --xgb-no-lag-residual `
  --xgb-lookup-history-mode legacy_structure_recent_level `
  --xgb-mic-select `
  --xgb-mic-top-n 20
```

Notes:

- `legacy_structure_recent_level` uses pre-2020 rows for seasonal lookup
  structure, while still scaling levels with the latest 12 months.
- MIC feature selection uses `minepy` when available. If `minepy` is not
  installed, the pipeline falls back to a deterministic quantile-binned mutual
  information approximation.

Key artifacts:

- `outputs/xgboost_best_params.json`: cached best parameters from Optuna
- `outputs/xgboost_optuna_trials.csv`: trial-level tuning results
- `outputs/xgboost_oof_predictions.csv`: raw walk-forward predictions for calibration
- `outputs/xgboost_oof_calibrated.csv`: calibrated out-of-fold predictions
- `submissions/submission_xgboost_top_aux_calibrated.csv`: calibrated XGBoost submission
- `submissions/submission.csv`: alias of the calibrated XGBoost submission
