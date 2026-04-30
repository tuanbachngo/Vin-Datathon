"""XGBoost tuning and walk-forward prediction helpers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    optuna = None

try:  # Package import when used from notebooks: from src.tuning import ...
    from .model import (
        TOP_AUX_FEATURES,
        XGBOOST_AUX_PARAMS,
        predict_xgboost_aux,
        train_xgboost_aux,
    )
    from .validation import Fold, default_folds, metrics
except ImportError:  # Direct script-style import when src/ is on sys.path.
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
    target_mode: str = "direct",
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None = None,
    outlier_downweight: bool = False,
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
            target_mode=target_mode,
            baseline_fn=baseline_fn,
            outlier_downweight=outlier_downweight,
            random_state=random_state,
        )
        pred = predict_xgboost_aux(
            model,
            val.Date,
            sales,
            fold.train_end,
            feature_order,
            selected_aux_features=selected_aux_features,
            drop_lag_features=drop_lag_features,
            target_mode=target_mode,
            baseline_fn=baseline_fn,
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
    search_mode: str = "narrow",
    base_params: dict[str, float | int | str] | None = None,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    drop_lag_features: bool = False,
    target_mode: str = "direct",
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None = None,
    outlier_downweight: bool = False,
    random_state: int = 42,
) -> XGBoostTuningResult:
    if optuna is None:
        raise ImportError(
            "optuna is not installed. Run `.venv\\Scripts\\python.exe -m pip install optuna`."
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = default_folds() if folds is None else folds
    fold_weights = np.asarray(
        [max(float(getattr(fold, "weight", 1.0)), 0.0) for fold in folds], dtype=float
    )
    if not np.isfinite(fold_weights).all() or fold_weights.sum() <= 0:
        fold_weights = np.ones(len(folds), dtype=float)
    anchor_params = dict(XGBOOST_AUX_PARAMS)
    if base_params is not None:
        anchor_params.update(base_params)

    def _clip_float(value: float, lo: float, hi: float) -> float:
        return float(min(max(value, lo), hi))

    def objective(trial: "optuna.trial.Trial") -> float:
        if search_mode == "narrow":
            lr0 = float(anchor_params.get("learning_rate", 0.03))
            depth0 = int(anchor_params.get("max_depth", 6))
            min_child0 = float(anchor_params.get("min_child_weight", 4.0))
            subsample0 = float(anchor_params.get("subsample", 0.9))
            colsample0 = float(anchor_params.get("colsample_bytree", 0.85))
            reg_lambda0 = float(anchor_params.get("reg_lambda", 1.0))

            trial_params = dict(anchor_params)
            trial_params.update(
                {
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        _clip_float(lr0 * 0.60, 0.005, 0.20),
                        _clip_float(lr0 * 1.50, 0.008, 0.25),
                        log=True,
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        max(3, depth0 - 2),
                        min(10, depth0 + 2),
                    ),
                    "min_child_weight": trial.suggest_float(
                        "min_child_weight",
                        _clip_float(min_child0 * 0.50, 0.5, 20.0),
                        _clip_float(min_child0 * 2.00, 1.0, 25.0),
                    ),
                    "subsample": trial.suggest_float(
                        "subsample",
                        _clip_float(subsample0 - 0.15, 0.60, 1.00),
                        _clip_float(subsample0 + 0.10, 0.65, 1.00),
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        _clip_float(colsample0 - 0.15, 0.60, 1.00),
                        _clip_float(colsample0 + 0.10, 0.65, 1.00),
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda",
                        _clip_float(reg_lambda0 * 0.50, 1e-3, 20.0),
                        _clip_float(reg_lambda0 * 2.50, 1e-2, 25.0),
                        log=True,
                    ),
                }
            )
        elif search_mode == "broad":
            trial_params = dict(anchor_params)
            trial_params.update(
                {
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
            )
        else:
            raise ValueError(f"Unsupported search_mode: {search_mode}")

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
                target_mode=target_mode,
                baseline_fn=baseline_fn,
                outlier_downweight=outlier_downweight,
                random_state=random_state,
            )
            pred = predict_xgboost_aux(
                model,
                val.Date,
                sales,
                fold.train_end,
                feature_order,
                selected_aux_features=selected_aux_features,
                drop_lag_features=drop_lag_features,
                target_mode=target_mode,
                baseline_fn=baseline_fn,
            )
            fold_metrics = metrics(val.Revenue.to_numpy(), pred)
            fold_rmses.append(fold_metrics["RMSE"])
            fold_maes.append(fold_metrics["MAE"])
            seen_weights = fold_weights[: len(fold_rmses)]
            partial_rmse = float(np.average(fold_rmses, weights=seen_weights))
            trial.report(partial_rmse, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        weighted_mae = float(np.average(fold_maes, weights=fold_weights))
        weighted_rmse = float(np.average(fold_rmses, weights=fold_weights))
        trial.set_user_attr("weighted_mae", weighted_mae)
        trial.set_user_attr("weighted_rmse", weighted_rmse)
        trial.set_user_attr("mean_mae", float(np.mean(fold_maes)))
        return weighted_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(anchor_params)
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
        row["weighted_mae"] = trial.user_attrs.get("weighted_mae")
        row["weighted_rmse"] = trial.user_attrs.get("weighted_rmse")
        trial_rows.append(row)

    return XGBoostTuningResult(
        best_params=best_params,
        best_value=float(study.best_value),
        trials=pd.DataFrame(trial_rows),
    )
