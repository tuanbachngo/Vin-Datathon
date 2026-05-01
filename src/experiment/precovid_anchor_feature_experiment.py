"""Pre-Covid anchor feature experiment for refined5-style XGBoost.

Adds features that reduce reliance on 2020-2022:
- 2017-2019 seasonal anchors by month / month x DOW / week / DOY / Tet bucket.
- long-memory 730/1095-day lag and rolling ratios.
- regime-gap features comparing recent/post-shock seasonal signal to the 2017-2019 anchor.

This script does not modify run_pipeline.py, model.py, features.py, or baselines.py.
Place it in src/ and run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .baselines import seasonal_residual_baseline
    from .model import XGBRegressor, XGBOOST_AUX_PARAMS, _aux_matrix, _filter_xgboost_feature_matrix
    from .run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        build_peak_month_error_reports,
        default_folds,
        fit_revenue_calibrator,
        load_sales,
        metrics,
    )
except ImportError:
    from baselines import seasonal_residual_baseline
    from model import XGBRegressor, XGBOOST_AUX_PARAMS, _aux_matrix, _filter_xgboost_feature_matrix
    from run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        build_peak_month_error_reports,
        default_folds,
        fit_revenue_calibrator,
        load_sales,
        metrics,
    )


REFINED5_PARAMS: dict[str, float | int | str] = {
    "objective": "reg:squarederror",
    "learning_rate": 0.025,
    "n_estimators": 650,
    "max_depth": 3,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.65,
    "gamma": 0.1,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}

TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29",
}
TET = {int(y): pd.Timestamp(d) for y, d in TET_DATES.items()}


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    feature_set: str
    baseline_mode: str
    weight_profile: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _clean_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())


def _parse_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _scale_label(scale: float) -> str:
    return f"x{int(round(scale * 100)):03d}"


def _safe_divide(num: np.ndarray, den: np.ndarray, *, default: float = 1.0) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, float(default), dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > 1e-9))
    return np.where(np.isfinite(out), out, float(default))


def _signed_days_to_tet(date: pd.Timestamp) -> int:
    date = pd.Timestamp(date)
    candidates = [TET.get(date.year - 1), TET.get(date.year), TET.get(date.year + 1)]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0
    diffs = [(date - c).days for c in candidates]
    return int(min(diffs, key=abs))


def _add_calendar_keys(frame: pd.DataFrame, *, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    d = pd.to_datetime(out[date_col])
    out["month"] = d.dt.month
    out["dow"] = d.dt.dayofweek
    out["week"] = d.dt.isocalendar().week.astype(int)
    out["doy"] = d.dt.dayofyear
    out["is_q4"] = out["month"].isin([10, 11, 12]).astype(int)
    out["month_end_window"] = (d.dt.days_in_month - d.dt.day <= 3).astype(int)
    days_to_tet = d.apply(_signed_days_to_tet)
    out["days_to_tet"] = days_to_tet
    out["pre_tet_window"] = ((days_to_tet >= -21) & (days_to_tet < 0)).astype(int)
    out["post_tet_window"] = ((days_to_tet > 0) & (days_to_tet <= 14)).astype(int)
    out["tet_week"] = (days_to_tet.abs() <= 7).astype(int)
    out["tet_bucket"] = np.select(
        [
            out["pre_tet_window"].astype(bool),
            out["post_tet_window"].astype(bool),
            out["tet_week"].astype(bool),
        ],
        ["pre_tet", "post_tet", "tet_week"],
        default="normal",
    )
    return out


def _stable_history(sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    hist = sales[sales.Date <= as_of].copy()
    stable = hist[(hist.Date.dt.year >= 2017) & (hist.Date.dt.year <= 2019)].copy()
    if len(stable) < 180:
        stable = hist[(hist.Date.dt.year >= 2016) & (hist.Date.dt.year <= 2019)].copy()
    if len(stable) < 180:
        stable = hist[hist.Date.dt.year <= 2019].copy()
    if len(stable) < 60:
        stable = hist.copy()
    return stable


def _recent_postshock_history(sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    hist = sales[sales.Date <= as_of].copy()
    recent = hist[(hist.Date.dt.year >= 2020) & (hist.Date.dt.year <= 2022)].copy()
    if len(recent) < 180:
        recent = hist[hist.Date > (pd.Timestamp(as_of) - pd.DateOffset(years=3))].copy()
    if len(recent) < 60:
        recent = hist.copy()
    return recent


def _lookup_from_history(history: pd.DataFrame, target_dates: pd.Series, *, prefix: str) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(target_dates)).reset_index(drop=True)
    hist = _add_calendar_keys(history[["Date", "Revenue"]].copy())
    target = _add_calendar_keys(pd.DataFrame({"Date": dates}))
    overall = float(hist["Revenue"].mean()) if len(hist) else 1.0
    if not np.isfinite(overall) or overall <= 0:
        overall = 1.0

    month_mean = hist.groupby("month").Revenue.mean()
    month_dow_mean = hist.groupby(["month", "dow"]).Revenue.mean()
    week_mean = hist.groupby("week").Revenue.mean()
    doy_mean = hist.groupby("doy").Revenue.mean()
    tet_mean = hist.groupby("tet_bucket").Revenue.mean()
    q4_mean = hist.groupby("is_q4").Revenue.mean()

    out = pd.DataFrame(index=range(len(target)))
    out[f"{prefix}_month_mean"] = target["month"].map(month_mean).fillna(overall).to_numpy(dtype=float)
    out[f"{prefix}_week_mean"] = target["week"].map(week_mean).fillna(out[f"{prefix}_month_mean"]).to_numpy(dtype=float)
    out[f"{prefix}_doy_mean"] = target["doy"].map(doy_mean).fillna(out[f"{prefix}_month_mean"]).to_numpy(dtype=float)
    out[f"{prefix}_month_dow_mean"] = [
        month_dow_mean.get((int(m), int(dw)), np.nan)
        for m, dw in zip(target["month"], target["dow"])
    ]
    out[f"{prefix}_month_dow_mean"] = pd.Series(out[f"{prefix}_month_dow_mean"]).fillna(out[f"{prefix}_month_mean"]).to_numpy(dtype=float)
    out[f"{prefix}_tet_bucket_mean"] = target["tet_bucket"].map(tet_mean).fillna(overall).to_numpy(dtype=float)
    out[f"{prefix}_q4_mean"] = target["is_q4"].map(q4_mean).fillna(overall).to_numpy(dtype=float)
    out[f"{prefix}_overall_mean"] = overall
    for col in list(out.columns):
        out[f"log_{col}"] = np.log(np.maximum(out[col].astype(float), 1.0))
    return out


def _long_memory_features(target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp, *, anchor_col: np.ndarray) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(target_dates)).reset_index(drop=True)
    hist = sales[sales.Date <= as_of].set_index("Date").sort_index()["Revenue"].astype(float)
    lag730 = hist.reindex(dates - pd.Timedelta(days=730)).to_numpy(dtype=float)
    lag1095 = hist.reindex(dates - pd.Timedelta(days=1095)).to_numpy(dtype=float)
    roll30 = hist.rolling(30, min_periods=10).mean()
    roll90 = hist.rolling(90, min_periods=30).mean()
    ewm90 = hist.ewm(span=90, min_periods=30).mean()
    roll30_730 = roll30.reindex(dates - pd.Timedelta(days=730)).to_numpy(dtype=float)
    roll30_1095 = roll30.reindex(dates - pd.Timedelta(days=1095)).to_numpy(dtype=float)
    roll90_730 = roll90.reindex(dates - pd.Timedelta(days=730)).to_numpy(dtype=float)
    roll90_1095 = roll90.reindex(dates - pd.Timedelta(days=1095)).to_numpy(dtype=float)
    ewm90_730 = ewm90.reindex(dates - pd.Timedelta(days=730)).to_numpy(dtype=float)
    ewm90_1095 = ewm90.reindex(dates - pd.Timedelta(days=1095)).to_numpy(dtype=float)
    anchor = np.asarray(anchor_col, dtype=float)

    out = pd.DataFrame({
        "pc_lag730": lag730,
        "pc_lag1095": lag1095,
        "pc_roll30_730": roll30_730,
        "pc_roll30_1095": roll30_1095,
        "pc_roll90_730": roll90_730,
        "pc_roll90_1095": roll90_1095,
        "pc_ewm90_730": ewm90_730,
        "pc_ewm90_1095": ewm90_1095,
    })
    out["pc_lag730_lag1095_ratio"] = np.clip(_safe_divide(lag730, lag1095), 0.3, 3.0)
    out["pc_roll30_730_1095_ratio"] = np.clip(_safe_divide(roll30_730, roll30_1095), 0.3, 3.0)
    out["pc_roll90_730_1095_ratio"] = np.clip(_safe_divide(roll90_730, roll90_1095), 0.3, 3.0)
    out["pc_ewm90_730_1095_ratio"] = np.clip(_safe_divide(ewm90_730, ewm90_1095), 0.3, 3.0)
    out["pc_lag730_anchor_ratio"] = np.clip(_safe_divide(lag730, anchor), 0.3, 3.0)
    out["pc_lag1095_anchor_ratio"] = np.clip(_safe_divide(lag1095, anchor), 0.3, 3.0)
    out["pc_roll30_730_anchor_ratio"] = np.clip(_safe_divide(roll30_730, anchor), 0.3, 3.0)
    out["pc_lag730_minus_anchor"] = lag730 - anchor
    out["pc_lag1095_minus_anchor"] = lag1095 - anchor
    out["pc_roll30_730_minus_anchor"] = roll30_730 - anchor
    for col in ["pc_lag730", "pc_lag1095", "pc_roll30_730", "pc_roll30_1095", "pc_roll90_730", "pc_roll90_1095", "pc_ewm90_730", "pc_ewm90_1095"]:
        out[f"log_{col}"] = np.log(np.maximum(out[col].astype(float), 1.0))
    return out


def _interaction_features(target_dates: pd.Series, anchor: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(target_dates)).reset_index(drop=True)
    cal = _add_calendar_keys(pd.DataFrame({"Date": dates}))
    base = anchor["anchor_2017_2019_month_dow_mean"].astype(float).to_numpy()
    return pd.DataFrame({
        "pc_anchor_x_month_end": base * cal["month_end_window"].to_numpy(dtype=float),
        "pc_anchor_x_q4": base * cal["is_q4"].to_numpy(dtype=float),
        "pc_anchor_x_pre_tet": base * cal["pre_tet_window"].to_numpy(dtype=float),
        "pc_anchor_x_post_tet": base * cal["post_tet_window"].to_numpy(dtype=float),
        "pc_anchor_x_tet_week": base * cal["tet_week"].to_numpy(dtype=float),
    })


def build_precovid_extra_features(dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp, *, feature_set: str) -> pd.DataFrame:
    stable = _stable_history(sales, as_of)
    recent = _recent_postshock_history(sales, as_of)
    anchor = _lookup_from_history(stable, dates, prefix="anchor_2017_2019")
    parts = [anchor]

    if feature_set in {"anchor_long", "anchor_gap", "anchor_interactions"}:
        parts.append(_long_memory_features(dates, sales, as_of, anchor_col=anchor["anchor_2017_2019_month_dow_mean"].to_numpy(dtype=float)))

    if feature_set in {"anchor_gap", "anchor_interactions"}:
        recent_lookup = _lookup_from_history(recent, dates, prefix="recent_2020_2022")
        stable_main = anchor["anchor_2017_2019_month_dow_mean"].to_numpy(dtype=float)
        recent_main = recent_lookup["recent_2020_2022_month_dow_mean"].to_numpy(dtype=float)
        gap = pd.DataFrame(index=range(len(anchor)))
        gap["regime_gap_recent_minus_anchor"] = recent_main - stable_main
        gap["regime_ratio_recent_anchor"] = np.clip(_safe_divide(recent_main, stable_main), 0.3, 3.0)
        gap["log_regime_ratio_recent_anchor"] = np.log(np.maximum(gap["regime_ratio_recent_anchor"].astype(float), 1e-6))
        gap["regime_gap_month_recent_minus_anchor"] = recent_lookup["recent_2020_2022_month_mean"].to_numpy(dtype=float) - anchor["anchor_2017_2019_month_mean"].to_numpy(dtype=float)
        gap["regime_ratio_month_recent_anchor"] = np.clip(_safe_divide(recent_lookup["recent_2020_2022_month_mean"].to_numpy(dtype=float), anchor["anchor_2017_2019_month_mean"].to_numpy(dtype=float)), 0.3, 3.0)
        parts.append(gap)

    if feature_set == "anchor_interactions":
        parts.append(_interaction_features(dates, anchor))

    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    # Clip extreme values so ratios/gaps do not dominate one split.
    for col in X.columns:
        if X[col].dtype.kind in "fc":
            s = X[col].astype(float)
            finite = s[np.isfinite(s)]
            if len(finite) > 20:
                lo, hi = finite.quantile([0.005, 0.995])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    X[col] = s.clip(lo, hi)
    return X.reset_index(drop=True)


def build_augmented_matrix(dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp, *, feature_set: str) -> pd.DataFrame:
    X_base = _aux_matrix(dates, sales, as_of, selected_aux_features=None)
    X_base = _filter_xgboost_feature_matrix(X_base, drop_lag_features=True)
    X_extra = build_precovid_extra_features(dates, sales, as_of, feature_set=feature_set)
    return pd.concat([X_base.reset_index(drop=True), X_extra.reset_index(drop=True)], axis=1)


def recovery_baseline(dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp, *, stable_weight: float) -> np.ndarray:
    current = np.asarray(seasonal_residual_baseline(dates, sales, as_of), dtype=float)
    stable = _lookup_from_history(_stable_history(sales, as_of), dates, prefix="tmp_anchor")["tmp_anchor_month_dow_mean"].to_numpy(dtype=float)
    out = float(stable_weight) * stable + (1.0 - float(stable_weight)) * current
    fallback = np.nanmedian(current) if np.isfinite(current).any() else np.nanmedian(stable)
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    return np.where(np.isfinite(out) & (out > 0), out, fallback)


def baseline_array(dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp, *, baseline_mode: str) -> np.ndarray:
    if baseline_mode == "default":
        b = np.asarray(seasonal_residual_baseline(dates, sales, as_of), dtype=float)
    elif baseline_mode.startswith("recovery"):
        suffix = baseline_mode.replace("recovery", "")
        stable_weight = float(suffix) / 10.0 if suffix else 0.7
        b = recovery_baseline(dates, sales, as_of, stable_weight=stable_weight)
    else:
        raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")
    fallback = np.nanmedian(b) if np.isfinite(b).any() else 1.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    return np.where(np.isfinite(b) & (b > 0), b, fallback)


def regime_sample_weight(dates: pd.Series, *, profile: str) -> np.ndarray | None:
    if profile == "none":
        return None
    years = pd.to_datetime(dates).dt.year.to_numpy()
    w = np.ones(len(years), dtype=float)
    if profile == "aggressive_w20_05":
        w[years <= 2016] = 0.5
        w[(years >= 2017) & (years <= 2019)] = 2.0
        w[(years >= 2020) & (years <= 2022)] = 0.5
    elif profile == "balanced_recovery":
        w[years <= 2016] = 0.7
        w[(years >= 2017) & (years <= 2019)] = 1.4
        w[(years >= 2020) & (years <= 2022)] = 1.0
    elif profile == "strong_recovery":
        w[years <= 2016] = 0.6
        w[(years >= 2017) & (years <= 2019)] = 1.7
        w[(years >= 2020) & (years <= 2022)] = 0.9
    else:
        raise ValueError(f"Unsupported weight_profile: {profile}")
    return w


def train_augmented_xgb(sales_train: pd.DataFrame, as_of: pd.Timestamp, *, config: ExperimentConfig, random_state: int = 42) -> tuple[object, list[str]]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed.")
    X = build_augmented_matrix(sales_train.Date, sales_train, as_of, feature_set=config.feature_set)
    y_log = np.log(np.maximum(sales_train.Revenue.to_numpy(dtype=float), 1.0))
    b = baseline_array(sales_train.Date, sales_train, as_of, baseline_mode=config.baseline_mode)
    valid = np.isfinite(b) & (b > 0)
    X = X.loc[valid].reset_index(drop=True)
    y = y_log[valid] - np.log(np.clip(b[valid], 1e-6, None))
    sample_weight = regime_sample_weight(sales_train.Date, profile=config.weight_profile)
    if sample_weight is not None:
        sample_weight = sample_weight[valid]
    model_params = dict(XGBOOST_AUX_PARAMS)
    model_params.update(REFINED5_PARAMS)
    model = XGBRegressor(**model_params, random_state=random_state)
    model.fit(X, y, sample_weight=sample_weight)
    return model, list(X.columns)


def predict_augmented_xgb(model: object, dates: pd.Series, sales_train: pd.DataFrame, as_of: pd.Timestamp, feature_order: list[str], *, config: ExperimentConfig) -> np.ndarray:
    X = build_augmented_matrix(dates, sales_train, as_of, feature_set=config.feature_set)
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_order]
    pred_resid = np.asarray(model.predict(X), dtype=float)
    b = baseline_array(dates, sales_train, as_of, baseline_mode=config.baseline_mode)
    return np.clip(np.clip(b, 1e-6, None) * np.exp(pred_resid), 0.0, None)


def collect_oof(sales: pd.DataFrame, *, folds: list, config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end].copy()
        val = sales[fold.mask_val(sales.Date)].sort_values("Date").reset_index(drop=True)
        model, feature_order = train_augmented_xgb(train, fold.train_end, config=config)
        pred = predict_augmented_xgb(model, val.Date, train, fold.train_end, feature_order, config=config)
        rows.append(pd.DataFrame({
            "config_id": config.config_id,
            "fold": fold.name,
            "train_end": fold.train_end,
            "Date": val.Date.to_numpy(),
            "actual_revenue": val.Revenue.to_numpy(dtype=float),
            "prediction_raw": pred,
        }))
    return pd.concat(rows, ignore_index=True)


def _weighted_summary(metric_rows: pd.DataFrame, folds: list) -> dict[str, float]:
    weight_map = {f.name: float(getattr(f, "weight", 1.0)) for f in folds}
    frame = metric_rows.copy()
    frame["weight"] = frame["fold"].map(weight_map).fillna(1.0)
    w = frame["weight"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for col in ["MAE", "RMSE", "R2", "MAPE"]:
        out[f"weighted_{col.lower()}"] = float(np.average(frame[col].to_numpy(dtype=float), weights=w))
    return out


def evaluate_oof(oof: pd.DataFrame, *, folds: list, no_calibration: bool, peak_month_quantile: float) -> tuple[pd.DataFrame, dict[str, Any], Any, pd.DataFrame]:
    frame = oof.copy()
    calibrator = None
    frame["prediction_calibrated"] = frame["prediction_raw"].to_numpy(dtype=float)
    if not no_calibration:
        cal_result = fit_revenue_calibrator(frame[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]])
        calibrator = cal_result.calibrator
        frame["prediction_calibrated"] = calibrator.predict(frame["prediction_raw"].to_numpy())
    rows: list[dict[str, Any]] = []
    for fold_name, part in frame.groupby("fold", sort=False):
        raw_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_raw"].to_numpy())
        cal_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_calibrated"].to_numpy())
        rows.append({"fold": fold_name, "prediction_type": "raw", **raw_m})
        rows.append({"fold": fold_name, "prediction_type": "calibrated", **cal_m})
    fold_metrics = pd.DataFrame(rows)
    raw_summary = _weighted_summary(fold_metrics[fold_metrics.prediction_type == "raw"], folds)
    cal_summary = _weighted_summary(fold_metrics[fold_metrics.prediction_type == "calibrated"], folds)
    fold3_raw = fold_metrics[(fold_metrics.fold == "fold3_test_proxy") & (fold_metrics.prediction_type == "raw")]
    fold3_cal = fold_metrics[(fold_metrics.fold == "fold3_test_proxy") & (fold_metrics.prediction_type == "calibrated")]
    _, peak_summary = build_peak_month_error_reports(frame[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]], calibrator=calibrator, quantile=float(peak_month_quantile))
    peak_all = peak_summary[(peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "peak_month")]
    non_peak_all = peak_summary[(peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "non_peak")]
    peak_row = peak_all.iloc[0] if not peak_all.empty else None
    non_peak_row = non_peak_all.iloc[0] if not non_peak_all.empty else None
    summary = {
        **{f"raw_{k}": v for k, v in raw_summary.items()},
        **{f"calibrated_{k}": v for k, v in cal_summary.items()},
        "raw_fold3_rmse": float(fold3_raw.iloc[0]["RMSE"]) if not fold3_raw.empty else np.nan,
        "calibrated_fold3_rmse": float(fold3_cal.iloc[0]["RMSE"]) if not fold3_cal.empty else np.nan,
        "raw_peak_month_ape": float(peak_row["avg_monthly_ape_raw"]) if peak_row is not None else np.nan,
        "calibrated_peak_month_ape": float(peak_row["avg_monthly_ape_calibrated"]) if peak_row is not None else np.nan,
        "raw_non_peak_ape": float(non_peak_row["avg_monthly_ape_raw"]) if non_peak_row is not None else np.nan,
        "calibrated_non_peak_ape": float(non_peak_row["avg_monthly_ape_calibrated"]) if non_peak_row is not None else np.nan,
    }
    return fold_metrics, summary, calibrator, frame


def predict_full_submission(sales: pd.DataFrame, *, config: ExperimentConfig, calibrator: Any, no_calibration: bool) -> pd.DataFrame:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = pd.Timestamp(sales.Date.max())
    model, feature_order = train_augmented_xgb(sales, as_of, config=config)
    pred_raw = predict_augmented_xgb(model, sub.Date, sales, as_of, feature_order, config=config)
    pred = pred_raw if no_calibration or calibrator is None else calibrator.predict(pred_raw)
    out = sub.copy()
    out["Revenue"] = np.clip(pred, 0.0, None)
    return out


def build_configs(feature_sets: list[str], baseline_modes: list[str], weight_profiles: list[str]) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    allowed = {"anchor", "anchor_long", "anchor_gap", "anchor_interactions"}
    for fs in feature_sets:
        if fs not in allowed:
            raise ValueError(f"Unsupported feature_set: {fs}")
        for bm in baseline_modes:
            for wp in weight_profiles:
                cid = _clean_id(f"{fs}_{bm}_{wp}")
                configs.append(ExperimentConfig(cid, fs, bm, wp))
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-Covid anchor feature experiment.")
    parser.add_argument("--feature-sets", type=str, default="anchor,anchor_long,anchor_gap")
    parser.add_argument("--baseline-modes", type=str, default="default")
    parser.add_argument("--weight-profiles", type=str, default="none")
    parser.add_argument("--cv-profile", choices=["late_priority", "legacy"], default="late_priority")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--peak-month-quantile", type=float, default=0.75)
    parser.add_argument("--scales", type=str, default="1.04,1.05,1.06")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output-prefix", type=str, default="submission_precovid_anchor")
    parser.add_argument("--export-all", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    sales = load_sales()
    folds = default_folds(profile=args.cv_profile)
    configs = build_configs(_parse_list(args.feature_sets), _parse_list(args.baseline_modes), _parse_list(args.weight_profiles))
    if args.max_configs and args.max_configs > 0:
        configs = configs[: int(args.max_configs)]
    print(f"Loaded sales: {len(sales)} rows, {sales.Date.min().date()} -> {sales.Date.max().date()}")
    print(f"Evaluating configs: {[c.config_id for c in configs]}")
    results: list[dict[str, Any]] = []
    all_fold_metrics: list[pd.DataFrame] = []
    calibrators: dict[str, Any] = {}
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}] {cfg.config_id}")
        oof = collect_oof(sales, folds=folds, config=cfg)
        fold_metrics, summary, calibrator, oof_eval = evaluate_oof(oof, folds=folds, no_calibration=bool(args.no_calibration), peak_month_quantile=float(args.peak_month_quantile))
        calibrators[cfg.config_id] = calibrator
        oof_path = OUT_DIR / f"precovid_anchor_oof_{cfg.config_id}.csv"
        oof_eval.to_csv(oof_path, index=False)
        fm = fold_metrics.copy()
        fm.insert(0, "config_id", cfg.config_id)
        all_fold_metrics.append(fm)
        row = {"config_id": cfg.config_id, "feature_set": cfg.feature_set, "baseline_mode": cfg.baseline_mode, "weight_profile": cfg.weight_profile, "oof_path": str(oof_path), **summary}
        results.append(row)
        print(json.dumps(_json_ready(row), indent=2, ensure_ascii=True))
    results_df = pd.DataFrame(results)
    score_col = "raw_weighted_rmse" if args.no_calibration else "calibrated_weighted_rmse"
    results_df = results_df.sort_values(score_col).reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "precovid_anchor_results.csv", index=False)
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(OUT_DIR / "precovid_anchor_cv_by_fold.csv", index=False)
    best_id = str(results_df.iloc[0]["config_id"])
    best_cfg = next(c for c in configs if c.config_id == best_id)
    best_payload = {"selected_config_id": best_id, "score_column": score_col, "selected_config": results_df.iloc[0].to_dict(), "all_configs": [c.__dict__ for c in configs], "params": REFINED5_PARAMS, "scales": _parse_float_list(args.scales), "no_calibration": bool(args.no_calibration)}
    with (OUT_DIR / "precovid_anchor_best_config.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready(best_payload), f, indent=2, ensure_ascii=True)
    export_configs = configs if args.export_all else [best_cfg]
    scale_values = _parse_float_list(args.scales)
    print("\nWriting full submissions...")
    for cfg in export_configs:
        out = predict_full_submission(sales, config=cfg, calibrator=calibrators.get(cfg.config_id), no_calibration=bool(args.no_calibration))
        base_path = sub_dir / f"{args.output_prefix}_{cfg.config_id}.csv"
        out.to_csv(base_path, index=False)
        print(f"Wrote {base_path}")
        for scale in scale_values:
            scaled = out.copy()
            scaled["Revenue"] = (scaled["Revenue"].astype(float) * float(scale)).clip(lower=0.0)
            scale_path = sub_dir / f"{args.output_prefix}_{cfg.config_id}_{_scale_label(scale)}.csv"
            scaled.to_csv(scale_path, index=False)
            print(f"Wrote {scale_path}")
    print("\nTop results:")
    cols = ["config_id", score_col, "calibrated_fold3_rmse", "calibrated_peak_month_ape", "raw_weighted_rmse", "raw_fold3_rmse", "raw_peak_month_ape"]
    print(results_df[[c for c in cols if c in results_df.columns]].head(10).to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'precovid_anchor_results.csv'}")
    print(f"Wrote {OUT_DIR / 'precovid_anchor_best_config.json'}")


if __name__ == "__main__":
    main()
