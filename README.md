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
