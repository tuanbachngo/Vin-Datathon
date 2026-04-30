"""Pre-Covid anchor feature pipeline helpers for refined5 residual XGBoost/LightGBM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

try:  # package style import
    from .baselines import seasonal_residual_baseline
    from .features import TET
    from .model import (
        XGBRegressor,
        _baseline_prediction_array,
        _build_xgboost_aux_matrix,
    )
except ImportError:  # script style import
    from baselines import seasonal_residual_baseline
    from features import TET
    from model import (
        XGBRegressor,
        _baseline_prediction_array,
        _build_xgboost_aux_matrix,
    )

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


REFINED5_PRECOVID_PARAMS: dict[str, float | int | str] = {
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

LIGHTGBM_PRECOVID_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": 750,
    "num_leaves": 20,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.8,
    "reg_lambda": 12.0,
    "n_jobs": 1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class PreCovidAnchorConfig:
    feature_set: str = "anchor_gap"  # anchor | anchor_long | anchor_gap
    regime_profile: str = "aggressive_w20_05"  # none | aggressive_w20_05 | balanced_recovery | strong_recovery
    baseline_mode: str = "default"  # default | recovery06 | recovery07 | recovery08
    model_kind: str = "xgb"  # xgb | lgbm
    random_state: int = 42


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > 1e-9))
    return out


def _signed_days_to_tet(date: pd.Timestamp) -> int:
    y = date.year
    candidates = [TET.get(y - 1), TET.get(y), TET.get(y + 1)]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0
    diffs = [(date - c).days for c in candidates]
    return int(min(diffs, key=abs))


def _tet_bucket(days_to_tet: int) -> str:
    if -21 <= days_to_tet < 0:
        return "pre_tet"
    if 0 <= days_to_tet <= 14:
        return "post_tet"
    if abs(days_to_tet) <= 45:
        return "near_tet"
    return "normal"


def _build_precovid_base(sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    hist = sales[sales.Date <= as_of].copy()
    hist = hist.sort_values("Date").reset_index(drop=True)
    base = hist[(hist.Date.dt.year >= 2017) & (hist.Date.dt.year <= 2019)].copy()
    if len(base) < 180:
        base = hist.copy()
    base["month"] = base.Date.dt.month
    base["dow"] = base.Date.dt.dayofweek
    base["days_to_tet"] = base.Date.map(_signed_days_to_tet)
    base["tet_bucket"] = base["days_to_tet"].map(_tet_bucket)
    base["is_q4"] = base["month"].isin([10, 11, 12]).astype(int)
    return base


def _precovid_anchor_lookup(
    target_dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    base = _build_precovid_base(sales, as_of)
    month_mean = base.groupby("month")["Revenue"].mean()
    month_dow_mean = base.groupby(["month", "dow"])["Revenue"].mean()
    tet_bucket_mean = base.groupby("tet_bucket")["Revenue"].mean()

    q4_month_base = base[base["month"].isin([10, 11, 12])]
    if len(q4_month_base):
        q4_month_mean = q4_month_base.groupby("month")["Revenue"].mean()
        q4_global = float(q4_month_base["Revenue"].mean())
    else:
        q4_month_mean = pd.Series(dtype=float)
        q4_global = float(base["Revenue"].mean())

    overall = float(base["Revenue"].mean()) if len(base) else 1.0
    if not np.isfinite(overall) or overall <= 0:
        overall = 1.0

    d = pd.DataFrame({"Date": pd.to_datetime(target_dates)})
    d["month"] = d.Date.dt.month
    d["dow"] = d.Date.dt.dayofweek
    d["days_to_tet"] = d.Date.map(_signed_days_to_tet)
    d["tet_bucket"] = d["days_to_tet"].map(_tet_bucket)

    anchor_month = d["month"].map(month_mean).fillna(overall).to_numpy(dtype=float)
    anchor_month_dow = d.apply(
        lambda r: month_dow_mean.get((r["month"], r["dow"]), np.nan), axis=1
    ).to_numpy(dtype=float)
    anchor_month_dow = np.where(np.isfinite(anchor_month_dow), anchor_month_dow, anchor_month)

    anchor_tet = d["tet_bucket"].map(tet_bucket_mean).fillna(overall).to_numpy(dtype=float)
    anchor_q4 = d["month"].map(q4_month_mean).fillna(q4_global).to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "anchor_2017_2019_month_mean": anchor_month,
            "anchor_2017_2019_month_dow_mean": anchor_month_dow,
            "anchor_2017_2019_tet_bucket_mean": anchor_tet,
            "anchor_2017_2019_q4_mean": anchor_q4,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _anchor_long_memory_features(
    target_dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    hist = sales[sales.Date <= as_of].set_index("Date").sort_index()["Revenue"]
    dates = pd.to_datetime(target_dates)
    v730 = hist.reindex(dates - pd.Timedelta(days=730)).to_numpy(dtype=float)
    v1095 = hist.reindex(dates - pd.Timedelta(days=1095)).to_numpy(dtype=float)
    mean_ = np.nanmean(np.vstack([v730, v1095]), axis=0)
    out = pd.DataFrame(
        {
            "anchor_long_memory_730": v730,
            "anchor_long_memory_1095": v1095,
            "anchor_long_memory_mean_730_1095": mean_,
            "anchor_long_memory_gap_730_1095": v730 - v1095,
            "anchor_long_memory_ratio_730_1095": _safe_div(v730, v1095),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _anchor_gap_features(
    X_base: pd.DataFrame,
    X_anchor: pd.DataFrame,
) -> pd.DataFrame:
    seasonal_month = (
        X_base["seasonal_month_mean"].to_numpy(dtype=float)
        if "seasonal_month_mean" in X_base.columns
        else np.full(len(X_base), np.nan, dtype=float)
    )
    seasonal_month_dow = (
        X_base["seasonal_month_dow_mean"].to_numpy(dtype=float)
        if "seasonal_month_dow_mean" in X_base.columns
        else seasonal_month
    )
    residual_anchor = (
        X_base["baseline_anchor"].to_numpy(dtype=float)
        if "baseline_anchor" in X_base.columns
        else seasonal_month
    )

    a_month = X_anchor["anchor_2017_2019_month_mean"].to_numpy(dtype=float)
    a_month_dow = X_anchor["anchor_2017_2019_month_dow_mean"].to_numpy(dtype=float)
    a_tet = X_anchor["anchor_2017_2019_tet_bucket_mean"].to_numpy(dtype=float)
    a_q4 = X_anchor["anchor_2017_2019_q4_mean"].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "anchor_gap_month_vs_lookup": a_month - seasonal_month,
            "anchor_gap_month_dow_vs_lookup": a_month_dow - seasonal_month_dow,
            "anchor_gap_tet_vs_lookup": a_tet - seasonal_month,
            "anchor_gap_q4_vs_lookup": a_q4 - seasonal_month,
            "anchor_gap_vs_residual_anchor": a_month_dow - residual_anchor,
            "anchor_gap_ratio_month_vs_lookup": _safe_div(a_month, seasonal_month),
            "anchor_gap_ratio_month_dow_vs_lookup": _safe_div(a_month_dow, seasonal_month_dow),
            "anchor_gap_ratio_vs_residual_anchor": _safe_div(a_month_dow, residual_anchor),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _recovery_baseline(
    target_dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    alpha: float,
) -> np.ndarray:
    default_anchor = seasonal_residual_baseline(target_dates, sales, as_of)
    precovid = _precovid_anchor_lookup(target_dates, sales, as_of)[
        "anchor_2017_2019_month_dow_mean"
    ].to_numpy(dtype=float)
    default_anchor = np.asarray(default_anchor, dtype=float)
    fallback = np.where(np.isfinite(default_anchor), default_anchor, precovid)
    precovid = np.where(np.isfinite(precovid), precovid, fallback)
    blended = float(alpha) * precovid + (1.0 - float(alpha)) * fallback
    return np.clip(blended, 1e-6, None)


def _resolve_baseline_fn(
    baseline_mode: str,
) -> Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None:
    if baseline_mode == "default":
        return None
    alpha_map = {
        "recovery06": 0.6,
        "recovery07": 0.7,
        "recovery08": 0.8,
    }
    if baseline_mode not in alpha_map:
        raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")
    alpha = float(alpha_map[baseline_mode])

    def _fn(
        target_dates: pd.Series,
        sales: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> np.ndarray:
        return _recovery_baseline(target_dates, sales, as_of, alpha=alpha)

    return _fn


def _regime_weights(
    dates: pd.Series,
    regime_profile: str,
) -> np.ndarray:
    years = pd.to_datetime(dates).dt.year.to_numpy(dtype=int)
    if regime_profile == "none":
        return np.ones(len(years), dtype=float)

    profiles = {
        "aggressive_w20_05": (0.5, 2.0, 0.5),
        "balanced_recovery": (0.7, 1.4, 1.0),
        "strong_recovery": (0.6, 1.7, 0.9),
    }
    if regime_profile not in profiles:
        raise ValueError(f"Unsupported regime_profile: {regime_profile}")
    w_12_16, w_17_19, w_20_22 = profiles[regime_profile]
    w = np.full(len(years), float(w_20_22), dtype=float)
    w = np.where(years <= 2016, float(w_12_16), w)
    w = np.where((years >= 2017) & (years <= 2019), float(w_17_19), w)
    w = np.where((years >= 2020) & (years <= 2022), float(w_20_22), w)
    return w


def _build_precovid_design_matrix(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    feature_set: str,
    baseline_mode: str,
) -> tuple[pd.DataFrame, Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None]:
    if feature_set not in {"anchor", "anchor_long", "anchor_gap"}:
        raise ValueError(f"Unsupported precovid feature_set: {feature_set}")
    baseline_fn = _resolve_baseline_fn(baseline_mode)
    X_base = _build_xgboost_aux_matrix(
        dates,
        sales,
        as_of,
        selected_aux_features=None,
        drop_lag_features=True,
        target_mode="residual",
        baseline_fn=baseline_fn,
    ).reset_index(drop=True)
    X_anchor = _precovid_anchor_lookup(dates, sales, as_of).reset_index(drop=True)

    parts = [X_base, X_anchor]
    if feature_set == "anchor_long":
        parts.append(_anchor_long_memory_features(dates, sales, as_of).reset_index(drop=True))
    elif feature_set == "anchor_gap":
        parts.append(_anchor_gap_features(X_base, X_anchor).reset_index(drop=True))

    X = pd.concat(parts, axis=1)
    # Add missing indicators for new anchor extensions.
    for col in X.columns:
        if col.startswith("anchor_") and X[col].isna().any():
            X[f"{col}_missing"] = X[col].isna().astype(np.int8)
    return X, baseline_fn


def _precovid_model_tag(model_kind: str) -> str:
    """Return a stable model tag used in logs and output filenames."""
    if model_kind == "xgb":
        return "xgboost"
    if model_kind == "lgbm":
        return "lightgbm"
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _make_precovid_regressor(
    *,
    cfg: PreCovidAnchorConfig,
    params: dict[str, float | int | str] | None,
):
    """Create the requested tree regressor.

    `params` is treated as an override only for XGBoost. The pipeline passes
    REFINED5_PRECOVID_PARAMS by default, and those keys are not LightGBM-compatible
    (`objective='reg:squarederror'`, `tree_method`, etc.). For LightGBM we use the
    dedicated LIGHTGBM_PRECOVID_PARAMS to avoid silently mixing parameter spaces.
    """
    if cfg.model_kind == "xgb":
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Run `.venv\Scripts\python.exe -m pip install xgboost`."
            )
        model_params = dict(REFINED5_PRECOVID_PARAMS)
        if params is not None:
            model_params.update(params)
        return XGBRegressor(**model_params, random_state=cfg.random_state)

    if cfg.model_kind == "lgbm":
        if LGBMRegressor is None:
            raise ImportError(
                "lightgbm is not installed. Run `.venv\Scripts\python.exe -m pip install lightgbm`."
            )
        model_params = dict(LIGHTGBM_PRECOVID_PARAMS)
        return LGBMRegressor(**model_params, random_state=cfg.random_state)

    raise ValueError(f"Unsupported model_kind: {cfg.model_kind}")


def train_precovid_anchor_xgb(
    sales_train: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    params: dict[str, float | int | str] | None,
    cfg: PreCovidAnchorConfig,
) -> tuple[object, list[str], Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None]:
    X_all, baseline_fn = _build_precovid_design_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        feature_set=cfg.feature_set,
        baseline_mode=cfg.baseline_mode,
    )
    y_all = np.log(sales_train.Revenue.to_numpy(dtype=float))
    baseline = _baseline_prediction_array(baseline_fn, sales_train.Date, sales_train, as_of)
    valid = np.isfinite(baseline)
    if not valid.any():
        raise RuntimeError("Residual baseline produced no valid rows in training.")

    X_fit = X_all.loc[valid].reset_index(drop=True)
    y_fit = y_all[valid] - np.log(np.clip(baseline[valid], 1e-6, None))
    sample_weight = _regime_weights(sales_train.Date, cfg.regime_profile)[valid]

    model = _make_precovid_regressor(cfg=cfg, params=params)
    model.fit(X_fit, y_fit, sample_weight=sample_weight)
    return model, list(X_fit.columns), baseline_fn


def predict_precovid_anchor_xgb(
    model: object,
    predict_dates: pd.Series,
    sales_context: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
    *,
    cfg: PreCovidAnchorConfig,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> np.ndarray:
    X_pred, _ = _build_precovid_design_matrix(
        predict_dates,
        sales_context,
        as_of,
        feature_set=cfg.feature_set,
        baseline_mode=cfg.baseline_mode,
    )
    for col in feature_order:
        if col not in X_pred.columns:
            X_pred[col] = np.nan
    X_pred = X_pred[feature_order]

    pred_log = np.asarray(model.predict(X_pred), dtype=float)
    baseline = _baseline_prediction_array(baseline_fn, predict_dates, sales_context, as_of)
    if np.isnan(baseline).any():
        raise RuntimeError("Residual baseline produced invalid rows in prediction.")
    return np.clip(baseline, 1e-6, None) * np.exp(pred_log)


def collect_precovid_anchor_oof_predictions(
    sales: pd.DataFrame,
    *,
    folds: list,
    params: dict[str, float | int | str] | None,
    cfg: PreCovidAnchorConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end].copy()
        val = sales[fold.mask_val(sales.Date)].sort_values("Date").reset_index(drop=True)
        model, feature_order, baseline_fn = train_precovid_anchor_xgb(
            train,
            as_of=fold.train_end,
            params=params,
            cfg=cfg,
        )
        pred = predict_precovid_anchor_xgb(
            model,
            val.Date,
            sales,
            fold.train_end,
            feature_order,
            cfg=cfg,
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


def predict_precovid_anchor_full(
    sales: pd.DataFrame,
    predict_dates: pd.Series,
    *,
    params: dict[str, float | int | str] | None,
    cfg: PreCovidAnchorConfig,
) -> np.ndarray:
    as_of = pd.Timestamp(sales["Date"].max())
    model, feature_order, baseline_fn = train_precovid_anchor_xgb(
        sales,
        as_of=as_of,
        params=params,
        cfg=cfg,
    )
    return predict_precovid_anchor_xgb(
        model,
        predict_dates,
        sales,
        as_of,
        feature_order,
        cfg=cfg,
        baseline_fn=baseline_fn,
    )

