"""XGBoost tuning and walk-forward prediction helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    optuna = None

from model import (
    TOP_AUX_FEATURES,
    XGBOOST_AUX_PARAMS,
    predict_xgboost_aux,
    train_xgboost_aux,
)
from validation import Fold, default_folds, metrics


@dataclass
class XGBoostTuningResult:
    best_params: dict[str, float | int | str]
    best_value: float
    trials: pd.DataFrame


def collect_xgboost_oof_predictions(
    sales: pd.DataFrame,
    *,
    params: dict[str, float | int | str] | None = None,
    folds: list[Fold] | None = None,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    drop_lag_features: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    folds = default_folds() if folds is None else folds
    rows: list[pd.DataFrame] = []

    for fold in folds:
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)]
        model, feature_order = train_xgboost_aux(
            train,
            as_of=fold.train_end,
            params=params,
            selected_aux_features=selected_aux_features,
            drop_lag_features=drop_lag_features,
            random_state=random_state,
        )
        pred = predict_xgboost_aux(
            model,
            val.Date,
            sales,
            fold.train_end,
            feature_order,
            selected_aux_features=selected_aux_features,
        )
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "Date": val.Date.to_numpy(),
                    "actual_revenue": val.Revenue.to_numpy(),
                    "prediction_raw": pred,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def tune_xgboost_hyperparameters(
    sales: pd.DataFrame,
    *,
    n_trials: int = 25,
    folds: list[Fold] | None = None,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    drop_lag_features: bool = False,
    random_state: int = 42,
) -> XGBoostTuningResult:
    if optuna is None:
        raise ImportError(
            "optuna is not installed. Run `.venv\\Scripts\\python.exe -m pip install optuna`."
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = default_folds() if folds is None else folds

    def objective(trial: "optuna.trial.Trial") -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }

        fold_rmses: list[float] = []
        fold_maes: list[float] = []
        for step, fold in enumerate(folds):
            train = sales[sales.Date <= fold.train_end]
            val = sales[fold.mask_val(sales.Date)]
            model, feature_order = train_xgboost_aux(
                train,
                as_of=fold.train_end,
                params=trial_params,
                selected_aux_features=selected_aux_features,
                drop_lag_features=drop_lag_features,
                random_state=random_state,
            )
            pred = predict_xgboost_aux(
                model,
                val.Date,
                sales,
                fold.train_end,
                feature_order,
                selected_aux_features=selected_aux_features,
            )
            fold_metrics = metrics(val.Revenue.to_numpy(), pred)
            fold_rmses.append(fold_metrics["RMSE"])
            fold_maes.append(fold_metrics["MAE"])
            trial.report(float(np.mean(fold_rmses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("mean_mae", float(np.mean(fold_maes)))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(XGBOOST_AUX_PARAMS)
    best_params.update(study.best_trial.params)

    trial_rows: list[dict[str, float | int | str | None]] = []
    for trial in study.trials:
        row: dict[str, float | int | str | None] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": float(trial.value) if trial.value is not None else None,
        }
        row.update(trial.params)
        row["mean_mae"] = trial.user_attrs.get("mean_mae")
        trial_rows.append(row)

    return XGBoostTuningResult(
        best_params=best_params,
        best_value=float(study.best_value),
        trials=pd.DataFrame(trial_rows),
    )
