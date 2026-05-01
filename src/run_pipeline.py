"""End-to-end driver: CV baselines + boosted ensembles, then write submissions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

try:  # Package import when used from notebooks: from src.run_pipeline import ...
    from .aux_features import configure_aux_batches, get_aux_batch_flags
    from .baselines import (
        seasonal_naive_growth_adjusted,
        seasonal_naive_last_year,
        seasonal_naive_mean_2y,
    )
    from .calibration import RevenueCalibrator, fit_revenue_calibrator
    from .model import (
        predict_gbr,
        predict_hist_gbm,
        predict_lightgbm,
        predict_lightgbm_aux,
        predict_mlp,
        predict_xgboost_aux,
        train_gbr,
        train_hist_gbm,
        train_lightgbm,
        train_lightgbm_aux,
        train_mlp,
        train_xgboost_aux,
    )
    from .tuning import collect_xgboost_oof_predictions, tune_xgboost_hyperparameters
    from .validation import default_folds, metrics
    from .precovid_anchor import (
        PreCovidAnchorConfig,
        REFINED5_PRECOVID_PARAMS,
        collect_precovid_anchor_oof_predictions,
        predict_precovid_anchor_full,
    )
except ImportError:  # Direct script-style import when running python src/run_pipeline.py.
    from aux_features import configure_aux_batches, get_aux_batch_flags
    from baselines import (
        seasonal_naive_growth_adjusted,
        seasonal_naive_last_year,
        seasonal_naive_mean_2y,
    )
    from calibration import RevenueCalibrator, fit_revenue_calibrator
    from model import (
        predict_gbr,
        predict_hist_gbm,
        predict_lightgbm,
        predict_lightgbm_aux,
        predict_mlp,
        predict_xgboost_aux,
        train_gbr,
        train_hist_gbm,
        train_lightgbm,
        train_lightgbm_aux,
        train_mlp,
        train_xgboost_aux,
    )
    from tuning import collect_xgboost_oof_predictions, tune_xgboost_hyperparameters
    from validation import default_folds, metrics
    from precovid_anchor import (
        PreCovidAnchorConfig,
        REFINED5_PRECOVID_PARAMS,
        collect_precovid_anchor_oof_predictions,
        predict_precovid_anchor_full,
    )

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"

OUT_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)


def xgb_artifact_paths(prefix: str) -> dict[str, Path]:
    return {
        "params": OUT_DIR / f"{prefix}_best_params.json",
        "trials": OUT_DIR / f"{prefix}_optuna_trials.csv",
        "oof": OUT_DIR / f"{prefix}_oof_predictions.csv",
        "oof_calibrated": OUT_DIR / f"{prefix}_oof_calibrated.csv",
        "calibration_curve": OUT_DIR / f"{prefix}_calibration_curve.csv",
        "calibration_summary": OUT_DIR / f"{prefix}_calibration_summary.json",
        "cv": OUT_DIR / f"{prefix}_cv_results.csv",
        "cv_weighted": OUT_DIR / f"{prefix}_cv_weighted_summary.csv",
        "peak_monthly": OUT_DIR / f"{prefix}_peak_month_errors.csv",
        "peak_summary": OUT_DIR / f"{prefix}_peak_month_error_summary.csv",
    }


def xgb_runtime_config(
    no_lag: bool,
    no_lag_residual: bool,
    outlier_downweight: bool,
) -> dict[str, str | bool]:
    if outlier_downweight:
        return {
            "artifact_prefix": "xgboost_outlier_downweight",
            "model_name": "xgboost_top_aux_outlier_downweight",
            "submission_prefix": "submission_xgboost_top_aux_outlier_downweight",
            "drop_lag_features": False,
            "target_mode": "direct",
            "outlier_downweight": True,
        }
    if no_lag_residual:
        return {
            "artifact_prefix": "xgboost_no_lag_residual",
            "model_name": "xgboost_top_aux_no_lag_residual",
            "submission_prefix": "submission_xgboost_top_aux_no_lag_residual",
            "drop_lag_features": True,
            "target_mode": "residual",
            "outlier_downweight": False,
        }
    if no_lag:
        return {
            "artifact_prefix": "xgboost_no_lag",
            "model_name": "xgboost_top_aux_no_lag",
            "submission_prefix": "submission_xgboost_top_aux_no_lag",
            "drop_lag_features": True,
            "target_mode": "direct",
            "outlier_downweight": False,
        }
    return {
        "artifact_prefix": "xgboost",
        "model_name": "xgboost_top_aux_log_target",
        "submission_prefix": "submission_xgboost_top_aux",
        "drop_lag_features": False,
        "target_mode": "direct",
        "outlier_downweight": False,
    }


def load_sales() -> pd.DataFrame:
    s = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    return s.sort_values("Date").reset_index(drop=True)


def _json_ready_dict(payload: dict[str, float | int | str]) -> dict[str, float | int | str]:
    normalized: dict[str, float | int | str] = {}
    for key, value in payload.items():
        normalized[key] = value.item() if hasattr(value, "item") else value
    return normalized


def load_xgboost_params(
    path: Path,
) -> dict[str, float | int | str] | None:
    candidates: list[Path] = []
    if path.exists():
        candidates.append(path)
    candidates.extend(
        sorted(
            path.parent.glob(f"{path.stem}_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
    )
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    with latest.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_xgboost_params(
    params: dict[str, float | int | str],
    path: Path,
) -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    ts_path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    with ts_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready_dict(params), handle, indent=2, ensure_ascii=True)
    return ts_path


def save_calibration_summary(
    summary: dict[str, float],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready_dict(summary), handle, indent=2, ensure_ascii=True)


def weighted_cv_summary(
    cv: pd.DataFrame,
    *,
    folds: list,
) -> pd.DataFrame:
    fold_weight_map = {fold.name: float(getattr(fold, "weight", 1.0)) for fold in folds}
    metric_cols = ["MAE", "RMSE", "R2", "MAPE"]
    rows: list[dict[str, float | str]] = []
    for model, frame in cv.groupby("model", sort=False):
        frame = frame.copy()
        frame["weight"] = frame["fold"].map(fold_weight_map).fillna(0.0)
        w = frame["weight"].to_numpy(dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(frame), dtype=float)
        row: dict[str, float | str] = {"model": model}
        for col in metric_cols:
            row[col] = float(np.average(frame[col].to_numpy(dtype=float), weights=w))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_peak_month_error_reports(
    xgb_oof: pd.DataFrame,
    *,
    calibrator: RevenueCalibrator | None,
    quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = xgb_oof.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["prediction_calibrated"] = (
        calibrator.predict(frame["prediction_raw"].to_numpy())
        if calibrator is not None
        else frame["prediction_raw"].to_numpy()
    )
    frame["month"] = frame["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        frame.groupby(["fold", "month"], as_index=False)
        .agg(
            actual_revenue=("actual_revenue", "sum"),
            pred_raw=("prediction_raw", "sum"),
            pred_calibrated=("prediction_calibrated", "sum"),
            n_days=("Date", "size"),
        )
    )
    monthly["abs_err_raw"] = (monthly["actual_revenue"] - monthly["pred_raw"]).abs()
    monthly["abs_err_calibrated"] = (
        monthly["actual_revenue"] - monthly["pred_calibrated"]
    ).abs()
    monthly["ape_raw"] = monthly["abs_err_raw"] / monthly["actual_revenue"].clip(lower=1.0)
    monthly["ape_calibrated"] = (
        monthly["abs_err_calibrated"] / monthly["actual_revenue"].clip(lower=1.0)
    )
    monthly["peak_threshold"] = monthly.groupby("fold")["actual_revenue"].transform(
        lambda s: s.quantile(quantile)
    )
    monthly["is_peak_month"] = (
        monthly["actual_revenue"] >= monthly["peak_threshold"]
    ).astype(int)
    monthly = monthly.drop(columns=["peak_threshold"])

    summary = (
        monthly.groupby(["fold", "is_peak_month"], as_index=False)
        .agg(
            months=("month", "size"),
            avg_monthly_abs_err_raw=("abs_err_raw", "mean"),
            avg_monthly_abs_err_calibrated=("abs_err_calibrated", "mean"),
            avg_monthly_ape_raw=("ape_raw", "mean"),
            avg_monthly_ape_calibrated=("ape_calibrated", "mean"),
            mean_actual_revenue=("actual_revenue", "mean"),
        )
    )
    overall = (
        monthly.groupby(["is_peak_month"], as_index=False)
        .agg(
            months=("month", "size"),
            avg_monthly_abs_err_raw=("abs_err_raw", "mean"),
            avg_monthly_abs_err_calibrated=("abs_err_calibrated", "mean"),
            avg_monthly_ape_raw=("ape_raw", "mean"),
            avg_monthly_ape_calibrated=("ape_calibrated", "mean"),
            mean_actual_revenue=("actual_revenue", "mean"),
        )
        .assign(fold="ALL")
    )
    summary = pd.concat([summary, overall], ignore_index=True)
    summary["segment"] = np.where(summary["is_peak_month"] == 1, "peak_month", "non_peak")
    summary = summary.drop(columns=["is_peak_month"]).sort_values(["fold", "segment"])
    return monthly, summary


def _predict_xgboost_strategy(
    *,
    sales_train: pd.DataFrame,
    predict_dates: pd.Series,
    sales_context: pd.DataFrame,
    as_of: pd.Timestamp,
    xgb_params: dict[str, float | int | str] | None,
    xgb_drop_lag_features: bool,
    xgb_target_mode: str,
    xgb_outlier_downweight: bool,
    xgb_dual_horizon: bool,
    dual_horizon_cutoff: int,
    early_lag_blend: bool,
    early_blend_end_day: int,
    lag_blend_weight: float,
    lag_blend_mode: str,
    tail_blend_weight: float,
    tail_blend_start_day: int,
    tail_blend_mode: str,
) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(predict_dates)).reset_index(drop=True)
    n = len(dates)
    if n == 0:
        return np.array([], dtype=float)

    day_idx = np.arange(1, n + 1)
    cutoff = int(max(1, min(int(dual_horizon_cutoff), n)))
    base_pred = np.full(n, np.nan, dtype=float)

    if xgb_dual_horizon:
        early_mask = day_idx <= cutoff
        late_mask = ~early_mask

        model_early, feat_early = train_xgboost_aux(
            sales_train,
            as_of=as_of,
            params=xgb_params,
            drop_lag_features=xgb_drop_lag_features,
            target_mode=xgb_target_mode,
            outlier_downweight=xgb_outlier_downweight,
        )
        if early_mask.any():
            base_pred[early_mask] = predict_xgboost_aux(
                model_early,
                dates.loc[early_mask],
                sales_context,
                as_of,
                feat_early,
                drop_lag_features=xgb_drop_lag_features,
                target_mode=xgb_target_mode,
            )

        if late_mask.any():
            model_late, feat_late = train_xgboost_aux(
                sales_train,
                as_of=as_of,
                params=xgb_params,
                drop_lag_features=True,
                target_mode="direct",
                outlier_downweight=xgb_outlier_downweight,
            )
            base_pred[late_mask] = predict_xgboost_aux(
                model_late,
                dates.loc[late_mask],
                sales_context,
                as_of,
                feat_late,
                drop_lag_features=True,
                target_mode="direct",
            )
    else:
        model, feat = train_xgboost_aux(
            sales_train,
            as_of=as_of,
            params=xgb_params,
            drop_lag_features=xgb_drop_lag_features,
            target_mode=xgb_target_mode,
            outlier_downweight=xgb_outlier_downweight,
        )
        base_pred = predict_xgboost_aux(
            model,
            dates,
            sales_context,
            as_of,
            feat,
            drop_lag_features=xgb_drop_lag_features,
            target_mode=xgb_target_mode,
        )

    # Optional early-horizon lag blend on top of base prediction.
    if early_lag_blend:
        early_w = float(np.clip(lag_blend_weight, 0.0, 1.0))
        early_end = int(max(1, min(int(early_blend_end_day), n)))
        early_mask = day_idx <= early_end
        if early_w > 0.0 and early_mask.any():
            if lag_blend_mode not in {"residual", "direct"}:
                raise ValueError(f"Unsupported lag_blend_mode: {lag_blend_mode}")
            model_lag, feat_lag = train_xgboost_aux(
                sales_train,
                as_of=as_of,
                params=xgb_params,
                drop_lag_features=False,
                target_mode=lag_blend_mode,
                outlier_downweight=xgb_outlier_downweight,
            )
            lag_pred = predict_xgboost_aux(
                model_lag,
                dates,
                sales_context,
                as_of,
                feat_lag,
                drop_lag_features=False,
                target_mode=lag_blend_mode,
            )
            base_pred = np.asarray(base_pred, dtype=float).copy()
            base_pred[early_mask] = (
                early_w * lag_pred[early_mask]
                + (1.0 - early_w) * base_pred[early_mask]
            )

    blend_w = float(np.clip(tail_blend_weight, 0.0, 1.0))
    if blend_w <= 0.0:
        return np.asarray(base_pred, dtype=float)

    blend_start = int(max(1, min(int(tail_blend_start_day), n)))
    tail_mask = day_idx >= blend_start
    if not tail_mask.any():
        return np.asarray(base_pred, dtype=float)

    if tail_blend_mode not in {"no_lag", "no_lag_residual"}:
        raise ValueError(f"Unsupported tail_blend_mode: {tail_blend_mode}")
    tail_target_mode = "residual" if tail_blend_mode == "no_lag_residual" else "direct"

    model_tail, feat_tail = train_xgboost_aux(
        sales_train,
        as_of=as_of,
        params=xgb_params,
        drop_lag_features=True,
        target_mode=tail_target_mode,
        outlier_downweight=xgb_outlier_downweight,
    )
    tail_pred = predict_xgboost_aux(
        model_tail,
        dates,
        sales_context,
        as_of,
        feat_tail,
        drop_lag_features=True,
        target_mode=tail_target_mode,
    )
    out = np.asarray(base_pred, dtype=float).copy()
    out[tail_mask] = (1.0 - blend_w) * out[tail_mask] + blend_w * tail_pred[tail_mask]
    return out


def collect_xgboost_oof_predictions_with_strategy(
    sales: pd.DataFrame,
    *,
    folds: list,
    xgb_params: dict[str, float | int | str] | None,
    xgb_drop_lag_features: bool,
    xgb_target_mode: str,
    xgb_outlier_downweight: bool,
    xgb_dual_horizon: bool,
    dual_horizon_cutoff: int,
    early_lag_blend: bool,
    early_blend_end_day: int,
    lag_blend_weight: float,
    lag_blend_mode: str,
    tail_blend_weight: float,
    tail_blend_start_day: int,
    tail_blend_mode: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)]
        pred = _predict_xgboost_strategy(
            sales_train=train,
            predict_dates=val.Date,
            sales_context=sales,
            as_of=fold.train_end,
            xgb_params=xgb_params,
            xgb_drop_lag_features=xgb_drop_lag_features,
            xgb_target_mode=xgb_target_mode,
            xgb_outlier_downweight=xgb_outlier_downweight,
            xgb_dual_horizon=xgb_dual_horizon,
            dual_horizon_cutoff=dual_horizon_cutoff,
            early_lag_blend=early_lag_blend,
            early_blend_end_day=early_blend_end_day,
            lag_blend_weight=lag_blend_weight,
            lag_blend_mode=lag_blend_mode,
            tail_blend_weight=tail_blend_weight,
            tail_blend_start_day=tail_blend_start_day,
            tail_blend_mode=tail_blend_mode,
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


def evaluate_cv(
    sales: pd.DataFrame,
    *,
    folds: list | None = None,
    xgb_params: dict[str, float | int | str] | None = None,
    xgb_oof: pd.DataFrame | None = None,
    xgb_calibrator: RevenueCalibrator | None = None,
    xgb_drop_lag_features: bool = False,
    xgb_target_mode: str = "direct",
    xgb_model_name: str = "xgboost_top_aux_log_target",
    xgb_outlier_downweight: bool = False,
    include_all_models: bool = True,
) -> pd.DataFrame:
    folds = default_folds() if folds is None else folds
    rows: list[dict[str, float | str]] = []
    xgb_by_fold = None
    if xgb_oof is not None:
        xgb_by_fold = {
            fold_name: frame.sort_values("Date").reset_index(drop=True)
            for fold_name, frame in xgb_oof.groupby("fold", sort=False)
        }
    for fold in folds:
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)]
        as_of = fold.train_end

        model_predictions: list[tuple[str, np.ndarray]] = []
        if include_all_models:
            p_sn = seasonal_naive_last_year(val.Date, sales, as_of)
            p_mn = seasonal_naive_mean_2y(val.Date, sales, as_of)
            p_gr = seasonal_naive_growth_adjusted(val.Date, sales, as_of)

            p_hist = None
            try:
                hist_model, hist_feature_order = train_hist_gbm(train, as_of=as_of)
                p_hist = predict_hist_gbm(
                    hist_model, val.Date, sales, as_of, hist_feature_order
                )
            except Exception as exc:
                print(f"[warn] Skipping hist_gbm_log_target on {fold.name}: {exc}")
            gbr_model, gbr_feature_order = train_gbr(train, as_of=as_of)
            lightgbm_model, lightgbm_feature_order = train_lightgbm(train, as_of=as_of)
            lightgbm_aux_model, lightgbm_aux_feature_order = train_lightgbm_aux(
                train, as_of=as_of
            )
            lightgbm_all_aux_model, lightgbm_all_aux_feature_order = train_lightgbm_aux(
                train, as_of=as_of, selected_aux_features=None
            )
            mlp_model, mlp_feature_order = train_mlp(train, as_of=as_of)
            p_gbr = predict_gbr(gbr_model, val.Date, sales, as_of, gbr_feature_order)
            p_lgbm = predict_lightgbm(
                lightgbm_model, val.Date, sales, as_of, lightgbm_feature_order
            )
            p_lgbm_aux = predict_lightgbm_aux(
                lightgbm_aux_model,
                val.Date,
                sales,
                as_of,
                lightgbm_aux_feature_order,
            )
            p_lgbm_all_aux = predict_lightgbm_aux(
                lightgbm_all_aux_model,
                val.Date,
                sales,
                as_of,
                lightgbm_all_aux_feature_order,
                selected_aux_features=None,
            )
            p_mlp = predict_mlp(mlp_model, val.Date, sales, as_of, mlp_feature_order)

            model_predictions.extend(
                [
                    ("seasonal_naive_lag365", p_sn),
                    ("seasonal_naive_mean2y", p_mn),
                    ("seasonal_naive_growth", p_gr),
                    ("gbr_log_target", p_gbr),
                    ("lightgbm_log_target", p_lgbm),
                    ("lightgbm_top_aux_log_target", p_lgbm_aux),
                    ("lightgbm_all_aux_log_target", p_lgbm_all_aux),
                    ("mlp_deep_learning", p_mlp),
                ]
            )
            if p_hist is not None:
                model_predictions.insert(3, ("hist_gbm_log_target", p_hist))

        if xgb_by_fold is None:
            xgb_model, xgb_feature_order = train_xgboost_aux(
                train,
                as_of=as_of,
                params=xgb_params,
                drop_lag_features=xgb_drop_lag_features,
                target_mode=xgb_target_mode,
                outlier_downweight=xgb_outlier_downweight,
            )
            p_xgb = predict_xgboost_aux(
                xgb_model,
                val.Date,
                sales,
                as_of,
                xgb_feature_order,
                drop_lag_features=xgb_drop_lag_features,
                target_mode=xgb_target_mode,
            )
        else:
            fold_xgb = xgb_by_fold.get(fold.name)
            if fold_xgb is None:
                raise ValueError(f"Missing XGBoost OOF predictions for fold {fold.name}")
            p_xgb = fold_xgb["prediction_raw"].to_numpy()
            if len(p_xgb) != len(val):
                raise ValueError(
                    f"Fold {fold.name} length mismatch: {len(p_xgb)} vs {len(val)}"
                )

        model_predictions.append((xgb_model_name, p_xgb))

        for name, pred in model_predictions:
            m = metrics(val.Revenue.values, pred)
            rows.append({"fold": fold.name, "model": name, **m})

        if xgb_calibrator is not None:
            p_xgb_calibrated = xgb_calibrator.predict(p_xgb)
            m = metrics(val.Revenue.values, p_xgb_calibrated)
            rows.append(
                {
                    "fold": fold.name,
                    "model": f"{xgb_model_name}_calibrated",
                    **m,
                }
            )

    return pd.DataFrame(rows)


def fit_and_submit(
    sales: pd.DataFrame,
    *,
    xgb_params: dict[str, float | int | str] | None = None,
    xgb_calibrator: RevenueCalibrator | None = None,
    xgb_drop_lag_features: bool = False,
    xgb_target_mode: str = "direct",
    xgb_submission_prefix: str = "submission_xgboost_top_aux",
    xgb_outlier_downweight: bool = False,
    write_submission_alias: bool = True,
    submission_tag: str | None = None,
    xgb_dual_horizon: bool = False,
    dual_horizon_cutoff: int = 365,
    early_lag_blend: bool = False,
    early_blend_end_day: int = 180,
    lag_blend_weight: float = 0.10,
    lag_blend_mode: str = "residual",
    tail_blend_weight: float = 0.0,
    tail_blend_start_day: int = 366,
    tail_blend_mode: str = "no_lag_residual",
) -> Path:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = sales.Date.max()
    tag_clean = ""
    if submission_tag and submission_tag.strip():
        tag_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", submission_tag.strip())
    suffix = f"_{tag_clean}" if tag_clean else ""

    pred_hist = None
    try:
        hist_model, hist_feature_order = train_hist_gbm(sales, as_of=as_of)
        pred_hist = predict_hist_gbm(
            hist_model, sub.Date, sales, as_of, hist_feature_order
        )
    except Exception as exc:
        print(f"[warn] Skipping hist_gbm_log_target submission: {exc}")
    gbr_model, gbr_feature_order = train_gbr(sales, as_of=as_of)
    lightgbm_model, lightgbm_feature_order = train_lightgbm(sales, as_of=as_of)
    lightgbm_aux_model, lightgbm_aux_feature_order = train_lightgbm_aux(
        sales, as_of=as_of
    )
    lightgbm_all_aux_model, lightgbm_all_aux_feature_order = train_lightgbm_aux(
        sales, as_of=as_of, selected_aux_features=None
    )
    mlp_model, mlp_feature_order = train_mlp(sales, as_of=as_of)

    pred_gbr = predict_gbr(gbr_model, sub.Date, sales, as_of, gbr_feature_order)
    pred_lgbm = predict_lightgbm(
        lightgbm_model, sub.Date, sales, as_of, lightgbm_feature_order
    )
    pred_lgbm_aux = predict_lightgbm_aux(
        lightgbm_aux_model,
        sub.Date,
        sales,
        as_of,
        lightgbm_aux_feature_order,
    )
    pred_lgbm_all_aux = predict_lightgbm_aux(
        lightgbm_all_aux_model,
        sub.Date,
        sales,
        as_of,
        lightgbm_all_aux_feature_order,
        selected_aux_features=None,
    )
    sorted_idx = sub.sort_values("Date").index.to_numpy()
    pred_sorted = _predict_xgboost_strategy(
        sales_train=sales,
        predict_dates=sub.loc[sorted_idx, "Date"],
        sales_context=sales,
        as_of=as_of,
        xgb_params=xgb_params,
        xgb_drop_lag_features=xgb_drop_lag_features,
        xgb_target_mode=xgb_target_mode,
        xgb_outlier_downweight=xgb_outlier_downweight,
        xgb_dual_horizon=xgb_dual_horizon,
        dual_horizon_cutoff=dual_horizon_cutoff,
        early_lag_blend=early_lag_blend,
        early_blend_end_day=early_blend_end_day,
        lag_blend_weight=lag_blend_weight,
        lag_blend_mode=lag_blend_mode,
        tail_blend_weight=tail_blend_weight,
        tail_blend_start_day=tail_blend_start_day,
        tail_blend_mode=tail_blend_mode,
    )
    pred_xgb = np.zeros(len(sub), dtype=float)
    pred_xgb[sorted_idx] = pred_sorted
    pred_xgb_calibrated = (
        xgb_calibrator.predict(pred_xgb) if xgb_calibrator is not None else pred_xgb
    )
    pred_mlp = predict_mlp(mlp_model, sub.Date, sales, as_of, mlp_feature_order)

    if pred_hist is not None:
        out_hist = sub.copy()
        out_hist["Revenue"] = pred_hist
        out_hist.to_csv(SUB_DIR / "submission_hist_gbm_v4.csv", index=False)

    out_gbr = sub.copy()
    out_gbr["Revenue"] = pred_gbr
    out_gbr.to_csv(SUB_DIR / "submission_gbr_v4.csv", index=False)

    out_lgbm = sub.copy()
    out_lgbm["Revenue"] = pred_lgbm
    out_lgbm.to_csv(SUB_DIR / "submission_lightgbm_v4.csv", index=False)

    out_lgbm_aux = sub.copy()
    out_lgbm_aux["Revenue"] = pred_lgbm_aux
    out_lgbm_aux.to_csv(SUB_DIR / "submission_lightgbm_top_aux_v5.csv", index=False)

    out_lgbm_all_aux = sub.copy()
    out_lgbm_all_aux["Revenue"] = pred_lgbm_all_aux
    out_lgbm_all_aux.to_csv(
        SUB_DIR / "submission_lightgbm_all_aux_v5.csv", index=False
    )

    out_xgb = sub.copy()
    out_xgb["Revenue"] = pred_xgb
    out_xgb.to_csv(SUB_DIR / f"{xgb_submission_prefix}_raw{suffix}.csv", index=False)

    out_xgb_calibrated = sub.copy()
    out_xgb_calibrated["Revenue"] = pred_xgb_calibrated
    out_xgb_calibrated.to_csv(
        SUB_DIR / f"{xgb_submission_prefix}_calibrated{suffix}.csv", index=False
    )

    out_mlp = sub.copy()
    out_mlp["Revenue"] = pred_mlp
    out_mlp.to_csv(SUB_DIR / "submission_mlp_v4.csv", index=False)

    if write_submission_alias:
        out_xgb_calibrated.to_csv(SUB_DIR / "submission.csv", index=False)
        return SUB_DIR / "submission.csv"
    return SUB_DIR / f"{xgb_submission_prefix}_calibrated{suffix}.csv"


def _run_final_precovid_anchor(
    *,
    sales: pd.DataFrame,
    folds: list,
    submission_tag: str,
    peak_month_quantile: float,
    cv_profile: str,
    feature_set: str,
    regime_profile: str,
    baseline_mode: str,
    submission_scale: float,
    model_kind: str,
) -> tuple[Path, Path]:
    tag_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", submission_tag.strip()) if submission_tag else ""
    suffix = f"_{tag_clean}" if tag_clean else ""
    model_tag = "xgboost" if model_kind == "xgb" else "lightgbm"
    profile_tag = f"{model_tag}_{feature_set}_{regime_profile}_{baseline_mode}"
    artifact_prefix = f"{model_tag}_precovid_anchor_{feature_set}_{regime_profile}_{baseline_mode}{suffix}"
    paths = xgb_artifact_paths(artifact_prefix)

    cfg = PreCovidAnchorConfig(
        feature_set=feature_set,
        regime_profile=regime_profile,
        baseline_mode=baseline_mode,
        model_kind=model_kind,
        random_state=42,
    )
    print(
        "Using final pre-Covid anchor mode: "
        f"model_kind={model_kind}, feature_set={feature_set}, "
        f"regime_profile={regime_profile}, baseline_mode={baseline_mode}, "
        f"submission_scale={submission_scale:.4f}"
    )

    xgb_oof = collect_precovid_anchor_oof_predictions(
        sales,
        folds=folds,
        params=REFINED5_PRECOVID_PARAMS,
        cfg=cfg,
    )
    xgb_oof.to_csv(paths["oof"], index=False)

    calibration_result = fit_revenue_calibrator(xgb_oof)
    calibration_result.calibrated_oof.to_csv(paths["oof_calibrated"], index=False)
    calibration_result.calibrator.to_frame().to_csv(paths["calibration_curve"], index=False)
    save_calibration_summary(calibration_result.summary, paths["calibration_summary"])
    print(
        "Calibration RMSE (grouped by Date): "
        f"{calibration_result.summary['raw_rmse']:.3f} -> "
        f"{calibration_result.summary['calibrated_rmse']:.3f}"
    )

    peak_monthly, peak_summary = build_peak_month_error_reports(
        xgb_oof,
        calibrator=calibration_result.calibrator,
        quantile=float(peak_month_quantile),
    )
    peak_monthly.to_csv(paths["peak_monthly"], index=False)
    peak_summary.to_csv(paths["peak_summary"], index=False)
    print("\n=== Peak-month error summary (pre-Covid anchor OOF) ===")
    print(peak_summary.round(4).to_string(index=False))

    cv = evaluate_cv(
        sales,
        folds=folds,
        xgb_oof=xgb_oof,
        xgb_calibrator=calibration_result.calibrator,
        xgb_drop_lag_features=True,
        xgb_target_mode="residual",
        xgb_model_name=f"{model_tag}_precovid_anchor_{feature_set}_{regime_profile}_{baseline_mode}",
        include_all_models=False,
    )
    cv_weighted = weighted_cv_summary(cv, folds=folds)
    cv.to_csv(paths["cv"], index=False)
    cv_weighted.to_csv(paths["cv_weighted"], index=False)
    print("\n=== CV summary (pre-Covid anchor) ===")
    print(cv_weighted.round(3).to_string(index=False))

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    pred_raw = predict_precovid_anchor_full(
        sales,
        sample_sub["Date"],
        params=REFINED5_PRECOVID_PARAMS,
        cfg=cfg,
    )
    pred_cal = calibration_result.calibrator.predict(pred_raw)
    pred_scaled = np.asarray(pred_cal, dtype=float) * float(submission_scale)

    cal_path = SUB_DIR / f"submission_{model_tag}_precovid_anchor_calibrated{suffix}.csv"
    scaled_path = SUB_DIR / f"submission_{model_tag}_precovid_anchor_scaled{suffix}.csv"

    out_cal = sample_sub.copy()
    out_cal["Revenue"] = pred_cal
    out_cal.to_csv(cal_path, index=False)

    out_scaled = sample_sub.copy()
    out_scaled["Revenue"] = pred_scaled
    out_scaled.to_csv(scaled_path, index=False)

    meta = {
        "mode": "final_precovid_anchor",
        "model_kind": model_kind,
        "cv_profile": cv_profile,
        "feature_set": feature_set,
        "regime_profile": regime_profile,
        "baseline_mode": baseline_mode,
        "submission_scale": float(submission_scale),
        "refined5_params": _json_ready_dict(REFINED5_PRECOVID_PARAMS),
        "submission_calibrated": str(cal_path),
        "submission_scaled": str(scaled_path),
    }
    save_calibration_summary(meta, OUT_DIR / f"{artifact_prefix}_run_meta.json")
    return cal_path, scaled_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run forecasting CV and create submissions."
    )
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Run Optuna tuning for the XGBoost top-aux pipeline before CV/submit.",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=25,
        help="Number of Optuna trials when --tune-xgboost is enabled.",
    )
    parser.add_argument(
        "--xgb-no-lag",
        action="store_true",
        help="Use the no-lag XGBoost feature space while keeping the direct log-revenue target.",
    )
    parser.add_argument(
        "--xgb-no-lag-residual",
        action="store_true",
        help="Use the no-lag XGBoost feature space and train on residuals over a seasonal baseline.",
    )
    parser.add_argument(
        "--xgb-outlier-downweight",
        action="store_true",
        help="Downweight robust revenue/gross-margin outlier days when fitting XGBoost.",
    )
    parser.add_argument(
        "--xgb-dual-horizon",
        action="store_true",
        help="Use dual-horizon XGBoost: early horizon uses main setup, late horizon uses no-lag direct setup.",
    )
    parser.add_argument(
        "--dual-horizon-cutoff",
        type=int,
        default=365,
        help="Last horizon day handled by the early model when dual-horizon is enabled.",
    )
    parser.add_argument(
        "--early-lag-blend",
        action="store_true",
        help="Blend a lag-enabled model only on early horizon days over the base XGBoost prediction.",
    )
    parser.add_argument(
        "--early-blend-end-day",
        type=int,
        default=180,
        help="Last horizon day where early lag blend is applied.",
    )
    parser.add_argument(
        "--lag-blend-weight",
        type=float,
        default=0.10,
        help="Blend weight (0-1) for lag-enabled model on early horizon.",
    )
    parser.add_argument(
        "--lag-blend-mode",
        choices=["residual", "direct"],
        default="residual",
        help="Target mode for lag-enabled early blend model.",
    )
    parser.add_argument(
        "--tail-blend-weight",
        type=float,
        default=0.0,
        help="Blend weight (0-1) for a no-lag tail model applied on late horizon days.",
    )
    parser.add_argument(
        "--tail-blend-start-day",
        type=int,
        default=366,
        help="First horizon day where tail blend is applied.",
    )
    parser.add_argument(
        "--tail-blend-mode",
        choices=["no_lag", "no_lag_residual"],
        default="no_lag_residual",
        help="Tail blend model type.",
    )
    parser.add_argument(
        "--cv-profile",
        choices=["late_priority", "legacy"],
        default="late_priority",
        help="Walk-forward split profile. `late_priority` weights recent folds higher.",
    )
    parser.add_argument(
        "--xgb-tune-search",
        choices=["narrow", "broad"],
        default="narrow",
        help="Hyperparameter search mode: `narrow` tunes around current best params.",
    )
    parser.add_argument(
        "--peak-month-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold used to define peak months in OOF error analysis.",
    )
    parser.add_argument(
        "--no-submission-alias",
        action="store_true",
        help="Do not overwrite submissions/submission.csv alias.",
    )
    parser.add_argument(
        "--submission-tag",
        type=str,
        default="",
        help="Optional tag appended to XGBoost submission filenames to avoid overwrite.",
    )
    parser.add_argument(
        "--final-precovid-anchor",
        action="store_true",
        help="Run the production pre-Covid anchor refined5 pipeline and write final calibrated/scaled submissions.",
    )
    parser.add_argument(
        "--precovid-model-kind",
        choices=["xgb", "lgbm"],
        default="xgb",
        help="Learner used by --final-precovid-anchor: XGBoost or LightGBM.",
    )
    parser.add_argument(
        "--precovid-feature-set",
        choices=["anchor", "anchor_long", "anchor_gap"],
        default="anchor_gap",
        help="Pre-Covid feature extension set used when --final-precovid-anchor is enabled.",
    )
    parser.add_argument(
        "--regime-profile",
        choices=["none", "aggressive_w20_05", "balanced_recovery", "strong_recovery"],
        default="aggressive_w20_05",
        help="Regime sample-weight profile used when --final-precovid-anchor is enabled.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["default", "recovery06", "recovery07", "recovery08"],
        default="default",
        help="Residual baseline mode used when --final-precovid-anchor is enabled.",
    )
    parser.add_argument(
        "--submission-scale",
        type=float,
        default=1.05,
        help="Post-calibration global scale factor for the final pre-Covid submission.",
    )
    parser.add_argument(
        "--aux-batch1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable aux feature Batch 1 (promotion dependency + margin dilution).",
    )
    parser.add_argument(
        "--aux-batch2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable aux feature Batch 2 (traffic quality).",
    )
    parser.add_argument(
        "--aux-batch3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable aux feature Batch 3 (trust/retention diagnostics).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    active_aux_flags = configure_aux_batches(
        enable_batch1=bool(args.aux_batch1),
        enable_batch2=bool(args.aux_batch2),
        enable_batch3=bool(args.aux_batch3),
    )
    sales = load_sales()
    print(
        f"Loaded sales: {len(sales)} rows, "
        f"{sales.Date.min().date()} -> {sales.Date.max().date()}"
    )

    xgb_config = xgb_runtime_config(
        args.xgb_no_lag,
        args.xgb_no_lag_residual,
        args.xgb_outlier_downweight,
    )
    batch_tag = (
        f"b1{int(active_aux_flags['batch1'])}"
        f"b2{int(active_aux_flags['batch2'])}"
        f"b3{int(active_aux_flags['batch3'])}"
    )
    if not all(active_aux_flags.values()):
        xgb_config["artifact_prefix"] = f"{xgb_config['artifact_prefix']}_{batch_tag}"
        xgb_config["submission_prefix"] = f"{xgb_config['submission_prefix']}_{batch_tag}"
        xgb_config["model_name"] = f"{xgb_config['model_name']}_{batch_tag}"

    tail_blend_weight = float(np.clip(args.tail_blend_weight, 0.0, 1.0))
    lag_blend_weight = float(np.clip(args.lag_blend_weight, 0.0, 1.0))
    use_xgb_strategy = (
        bool(args.xgb_dual_horizon)
        or bool(args.early_lag_blend)
        or tail_blend_weight > 0.0
    )
    folds = default_folds(profile=args.cv_profile)
    fold_desc = ", ".join(f"{f.name}(w={f.weight:.2f})" for f in folds)
    print(f"Using CV profile `{args.cv_profile}`: {fold_desc}")
    print(f"Aux batch flags: {get_aux_batch_flags()} (tag={batch_tag})")
    if args.final_precovid_anchor:
        if args.tune_xgboost:
            print("[warn] --tune-xgboost is ignored in --final-precovid-anchor mode.")
        cal_path, scaled_path = _run_final_precovid_anchor(
            sales=sales,
            folds=folds,
            submission_tag=args.submission_tag,
            peak_month_quantile=float(args.peak_month_quantile),
            cv_profile=str(args.cv_profile),
            feature_set=str(args.precovid_feature_set),
            regime_profile=str(args.regime_profile),
            baseline_mode=str(args.baseline_mode),
            submission_scale=float(args.submission_scale),
            model_kind=str(args.precovid_model_kind),
        )
        print(f"wrote {cal_path}")
        print(f"wrote {scaled_path}")
        return

    if args.early_lag_blend and not args.xgb_no_lag_residual:
        print(
            "[warn] --early-lag-blend is recommended with --xgb-no-lag-residual "
            "so base prediction is full no-lag residual."
        )
    if use_xgb_strategy:
        print(
            "Using XGBoost strategy: "
            f"dual_horizon={args.xgb_dual_horizon}, "
            f"cutoff={args.dual_horizon_cutoff}, "
            f"early_lag_blend={args.early_lag_blend}, "
            f"early_end={args.early_blend_end_day}, "
            f"lag_weight={lag_blend_weight:.3f}, "
            f"lag_mode={args.lag_blend_mode}, "
            f"tail_blend_weight={tail_blend_weight:.3f}, "
            f"tail_blend_start_day={args.tail_blend_start_day}, "
            f"tail_blend_mode={args.tail_blend_mode}"
        )
    xgb_paths = xgb_artifact_paths(str(xgb_config["artifact_prefix"]))
    xgb_params = load_xgboost_params(xgb_paths["params"])
    xgb_source = "defaults"

    if args.tune_xgboost:
        print(
            "\n=== Optuna tuning XGBoost "
            f"({xgb_config['artifact_prefix']}, {args.xgb_trials} trials) ==="
        )
        tuning_result = tune_xgboost_hyperparameters(
            sales,
            n_trials=args.xgb_trials,
            folds=folds,
            search_mode=args.xgb_tune_search,
            base_params=xgb_params,
            drop_lag_features=bool(xgb_config["drop_lag_features"]),
            target_mode=str(xgb_config["target_mode"]),
            outlier_downweight=bool(xgb_config["outlier_downweight"]),
        )
        xgb_params = tuning_result.best_params
        xgb_source = f"optuna_{args.xgb_trials}_trials"
        saved_params_path = save_xgboost_params(xgb_params, xgb_paths["params"])
        tuning_result.trials.to_csv(xgb_paths["trials"], index=False)
        print(f"Best RMSE: {tuning_result.best_value:.3f}")
        print(f"Saved params to {saved_params_path}")
    elif xgb_params is not None:
        xgb_source = "latest_saved_json"

    print(f"\n=== XGBoost parameter source: {xgb_source} ===")
    if use_xgb_strategy:
        xgb_oof = collect_xgboost_oof_predictions_with_strategy(
            sales,
            folds=folds,
            xgb_params=xgb_params,
            xgb_drop_lag_features=bool(xgb_config["drop_lag_features"]),
            xgb_target_mode=str(xgb_config["target_mode"]),
            xgb_outlier_downweight=bool(xgb_config["outlier_downweight"]),
            xgb_dual_horizon=bool(args.xgb_dual_horizon),
            dual_horizon_cutoff=int(args.dual_horizon_cutoff),
            early_lag_blend=bool(args.early_lag_blend),
            early_blend_end_day=int(args.early_blend_end_day),
            lag_blend_weight=lag_blend_weight,
            lag_blend_mode=str(args.lag_blend_mode),
            tail_blend_weight=tail_blend_weight,
            tail_blend_start_day=int(args.tail_blend_start_day),
            tail_blend_mode=str(args.tail_blend_mode),
        )
    else:
        xgb_oof = collect_xgboost_oof_predictions(
            sales,
            params=xgb_params,
            folds=folds,
            drop_lag_features=bool(xgb_config["drop_lag_features"]),
            target_mode=str(xgb_config["target_mode"]),
            outlier_downweight=bool(xgb_config["outlier_downweight"]),
        )
    xgb_oof.to_csv(xgb_paths["oof"], index=False)

    calibration_result = fit_revenue_calibrator(xgb_oof)
    calibration_result.calibrated_oof.to_csv(xgb_paths["oof_calibrated"], index=False)
    calibration_result.calibrator.to_frame().to_csv(
        xgb_paths["calibration_curve"], index=False
    )
    save_calibration_summary(calibration_result.summary, xgb_paths["calibration_summary"])
    print(
        "Calibration RMSE (grouped by Date): "
        f"{calibration_result.summary['raw_rmse']:.3f} -> "
        f"{calibration_result.summary['calibrated_rmse']:.3f}"
    )

    peak_monthly, peak_summary = build_peak_month_error_reports(
        xgb_oof,
        calibrator=calibration_result.calibrator,
        quantile=float(args.peak_month_quantile),
    )
    peak_monthly.to_csv(xgb_paths["peak_monthly"], index=False)
    peak_summary.to_csv(xgb_paths["peak_summary"], index=False)
    print("\n=== Peak-month error summary (XGBoost OOF) ===")
    print(peak_summary.round(4).to_string(index=False))

    print("\n=== CV on configured walk-forward folds ===")
    cv = evaluate_cv(
        sales,
        folds=folds,
        xgb_params=xgb_params,
        xgb_oof=xgb_oof,
        xgb_calibrator=calibration_result.calibrator,
        xgb_drop_lag_features=bool(xgb_config["drop_lag_features"]),
        xgb_target_mode=str(xgb_config["target_mode"]),
        xgb_model_name=str(xgb_config["model_name"]),
        xgb_outlier_downweight=bool(xgb_config["outlier_downweight"]),
    )
    print(cv.round(3).to_string(index=False))

    print("\n=== CV summary by model (weighted by fold profile) ===")
    cv_weighted = weighted_cv_summary(cv, folds=folds)
    print(cv_weighted.round(3).to_string(index=False))
    cv.to_csv(xgb_paths["cv"], index=False)
    cv_weighted.to_csv(xgb_paths["cv_weighted"], index=False)

    print("\n=== Fit on all train, write submission ===")
    path = fit_and_submit(
        sales,
        xgb_params=xgb_params,
        xgb_calibrator=calibration_result.calibrator,
        xgb_drop_lag_features=bool(xgb_config["drop_lag_features"]),
        xgb_target_mode=str(xgb_config["target_mode"]),
        xgb_submission_prefix=str(xgb_config["submission_prefix"]),
        xgb_outlier_downweight=bool(xgb_config["outlier_downweight"]),
        submission_tag=args.submission_tag,
        xgb_dual_horizon=bool(args.xgb_dual_horizon),
        dual_horizon_cutoff=int(args.dual_horizon_cutoff),
        early_lag_blend=bool(args.early_lag_blend),
        early_blend_end_day=int(args.early_blend_end_day),
        lag_blend_weight=lag_blend_weight,
        lag_blend_mode=str(args.lag_blend_mode),
        tail_blend_weight=tail_blend_weight,
        tail_blend_start_day=int(args.tail_blend_start_day),
        tail_blend_mode=str(args.tail_blend_mode),
        write_submission_alias=not (
            args.no_submission_alias
            or args.xgb_no_lag_residual
            or args.xgb_no_lag
            or args.xgb_outlier_downweight
        ),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
