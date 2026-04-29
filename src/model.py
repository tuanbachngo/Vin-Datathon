"""Boosted tree forecasters on log(Revenue)."""
from __future__ import annotations
import os
from collections.abc import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    XGBRegressor = None

from baselines import (
    seasonal_lookup_level_adjusted,
    seasonal_naive_growth_adjusted,
    seasonal_naive_mean_2y,
    seasonal_residual_baseline,
)
from features import build_feature_matrix
from aux_features import (
    aux_feature_groups,
    build_aux_daily,
    build_aux_feature_matrix,
)


FEATURE_DROP = ["Date"]

HIST_GBM_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 800,
    "max_depth": None,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
}

SMOOTH_GBR_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.02,
    "n_estimators": 1000,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "subsample": 1.0,
}

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": 600,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "n_jobs": 1,
    "verbosity": -1,
}

LIGHTGBM_AUX_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.04,
    "n_estimators": 350,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "verbosity": -1,
}

XGBOOST_AUX_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "n_estimators": 700,
    "max_depth": 6,
    "min_child_weight": 4.0,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}

# Top auxiliary features selected from XGBoost gain importance averaged across
# all three CV folds on the current feature set (trend+seasonal + EWMA).
TOP_AUX_FEATURES = [
    "aux_unique_customers_trend",
    "aux_order_count_trend",
    "aux_sessions_trend",
    "aux_units_year1",
    "aux_unique_customers_year1",
    "aux_order_count_year1",
    "aux_payment_value_trend_month",
    "aux_refund_amount_trend_month",
    "aux_conversion_order_per_session_ewm90_548",
    "aux_avg_promo_discount_value_year2",
    "aux_cat_share_casual_ewm90_548",
    "aux_mean_days_of_supply_ewm30_730",
    "aux_conversion_order_per_session_ewm90_730",
    "aux_page_views_ewm90_548",
    "aux_refund_amount_trend_month_dow",
    "aux_payment_value_year1",
    "aux_net_item_sales_year1",
    "aux_sessions_ewm90_548",
    "aux_page_views_trend",
    "aux_unique_customers_roll30_548",
]

XGB_AUX_LAG_TOKENS = (
    "_lag548",
    "_lag730",
    "_year1",
    "_year2",
    "_roll30_",
    "_ewm30_",
    "_ewm90_",
    "_miss",
)

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-3,
    "learning_rate_init": 5e-4,
    "batch_size": 64,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "max_iter": 1500,
    "n_iter_no_change": 80,
}


def _add_missing_indicators(
    X: pd.DataFrame, *, indicator_columns: list[str] | None = None
) -> pd.DataFrame:
    X = X.copy()
    if indicator_columns is None:
        indicator_columns = [c for c in X.columns if X[c].isna().any()]
    for col in indicator_columns:
        X[f"{col}_missing"] = X[col].isna().astype(int)
    return X


def _feature_matrix(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    indicator_columns: list[str] | None = None,
) -> pd.DataFrame:
    feats = build_feature_matrix(dates, sales, as_of=as_of)
    X = feats.drop(columns=FEATURE_DROP)
    return _add_missing_indicators(X, indicator_columns=indicator_columns)


def _xy(
    frame: pd.DataFrame,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    X = _feature_matrix(frame.Date, sales, as_of=as_of)
    y = np.log(frame.Revenue.values) if "Revenue" in frame.columns else None
    return X, y


def _align_prediction_matrix(
    dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> pd.DataFrame:
    indicator_columns = [c[:-8] for c in feature_order if c.endswith("_missing")]
    X = _feature_matrix(
        dates, sales_train, as_of=as_of, indicator_columns=indicator_columns
    )
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_order]


def _aux_matrix(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    selected_aux_features: list[str] | None,
) -> pd.DataFrame:
    aux_daily = build_aux_daily()
    raw_aux_columns = list(dict.fromkeys(aux_feature_groups(aux_daily)["all_aux"]))
    X_base = _feature_matrix(dates, sales, as_of=as_of)
    X_aux = build_aux_feature_matrix(
        dates,
        as_of,
        raw_aux_columns,
        aux_daily=aux_daily,
    )
    if selected_aux_features is not None:
        keep = [c for c in selected_aux_features if c in X_aux.columns]
        X_aux = X_aux[keep]
    return pd.concat([X_base.reset_index(drop=True), X_aux.reset_index(drop=True)], axis=1)


def _align_aux_prediction_matrix(
    dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
    *,
    selected_aux_features: list[str] | None,
) -> pd.DataFrame:
    X = _aux_matrix(
        dates,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_order]


def _filter_xgboost_feature_matrix(
    X: pd.DataFrame,
    *,
    drop_lag_features: bool,
) -> pd.DataFrame:
    if not drop_lag_features:
        return X

    keep_columns: list[str] = []
    for col in X.columns:
        if col.startswith("aux_"):
            if not any(token in col for token in XGB_AUX_LAG_TOKENS):
                keep_columns.append(col)
            continue
        if col.startswith(("rev_lag_", "log_rev_lag_", "rev_roll", "rev_ewm")):
            continue
        keep_columns.append(col)
    return X[keep_columns]


def _baseline_prediction_array(
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
) -> np.ndarray:
    if baseline_fn is None:
        baseline_fn = seasonal_residual_baseline
    baseline = np.asarray(baseline_fn(dates, sales, as_of), dtype=float)
    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)
    return baseline


def _baseline_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _baseline_feature_frame(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    baseline_mean2y = _baseline_prediction_array(
        seasonal_naive_mean_2y, dates, sales, as_of
    )
    baseline_growth = _baseline_prediction_array(
        seasonal_naive_growth_adjusted, dates, sales, as_of
    )
    baseline_lookup = _baseline_prediction_array(
        seasonal_lookup_level_adjusted, dates, sales, as_of
    )
    baseline_anchor = _baseline_prediction_array(
        baseline_fn, dates, sales, as_of
    )

    baseline_stack = pd.DataFrame(
        {
            "baseline_mean2y": baseline_mean2y,
            "baseline_growth": baseline_growth,
            "baseline_lookup": baseline_lookup,
        }
    )
    consensus = baseline_stack.mean(axis=1, skipna=True).to_numpy(dtype=float)
    spread = (
        baseline_stack.max(axis=1, skipna=True) - baseline_stack.min(axis=1, skipna=True)
    ).to_numpy(dtype=float)

    features = pd.DataFrame(
        {
            "baseline_mean2y": baseline_mean2y,
            "baseline_growth": baseline_growth,
            "baseline_lookup": baseline_lookup,
            "baseline_anchor": baseline_anchor,
            "baseline_consensus": consensus,
            "baseline_spread": spread,
        }
    )

    for col in [
        "baseline_mean2y",
        "baseline_growth",
        "baseline_lookup",
        "baseline_anchor",
        "baseline_consensus",
    ]:
        features[f"log_{col}"] = np.log(np.clip(features[col].to_numpy(dtype=float), 1e-6, None))

    features["baseline_gap_lookup_mean2y"] = (
        features["baseline_lookup"] - features["baseline_mean2y"]
    )
    features["baseline_gap_growth_mean2y"] = (
        features["baseline_growth"] - features["baseline_mean2y"]
    )
    features["baseline_gap_lookup_growth"] = (
        features["baseline_lookup"] - features["baseline_growth"]
    )
    features["baseline_ratio_lookup_mean2y"] = _baseline_ratio(
        features["baseline_lookup"].to_numpy(),
        features["baseline_mean2y"].to_numpy(),
    )
    features["baseline_ratio_growth_mean2y"] = _baseline_ratio(
        features["baseline_growth"].to_numpy(),
        features["baseline_mean2y"].to_numpy(),
    )
    features["baseline_ratio_lookup_growth"] = _baseline_ratio(
        features["baseline_lookup"].to_numpy(),
        features["baseline_growth"].to_numpy(),
    )
    return features


def _augment_xgboost_feature_matrix(
    X: pd.DataFrame,
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    target_mode: str,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    if target_mode != "residual":
        return X
    baseline_features = _baseline_feature_frame(
        dates,
        sales,
        as_of,
        baseline_fn=baseline_fn,
    )
    return pd.concat([X.reset_index(drop=True), baseline_features.reset_index(drop=True)], axis=1)


def train_hist_gbm(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    random_state: int = 42,
) -> tuple[HistGradientBoostingRegressor, list[str]]:
    X, y = _xy(sales_train, sales_train, as_of=as_of)
    model = HistGradientBoostingRegressor(
        **HIST_GBM_PARAMS,
        random_state=random_state,
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_hist_gbm(
    model: HistGradientBoostingRegressor,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> np.ndarray:
    X = _align_prediction_matrix(
        val_dates, sales_train, as_of=as_of, feature_order=feature_order
    )
    return np.exp(model.predict(X))


def train_gbr(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    random_state: int = 42,
) -> tuple[Pipeline, list[str]]:
    X, y = _xy(sales_train, sales_train, as_of=as_of)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingRegressor(
                    **SMOOTH_GBR_PARAMS,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_gbr(
    model: Pipeline,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> np.ndarray:
    X = _align_prediction_matrix(
        val_dates, sales_train, as_of=as_of, feature_order=feature_order
    )
    return np.exp(model.predict(X))


def train_lightgbm(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    random_state: int = 42,
) -> tuple[object, list[str]]:
    if LGBMRegressor is None:
        raise ImportError(
            "lightgbm is not installed. Run `.venv\\Scripts\\python.exe -m pip install lightgbm`."
        )
    X, y = _xy(sales_train, sales_train, as_of=as_of)
    model = LGBMRegressor(
        **LIGHTGBM_PARAMS,
        random_state=random_state,
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_lightgbm(
    model: object,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> np.ndarray:
    X = _align_prediction_matrix(
        val_dates, sales_train, as_of=as_of, feature_order=feature_order
    )
    return np.exp(model.predict(X))


def train_lightgbm_aux(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    random_state: int = 42,
) -> tuple[object, list[str]]:
    if LGBMRegressor is None:
        raise ImportError(
            "lightgbm is not installed. Run `.venv\\Scripts\\python.exe -m pip install lightgbm`."
        )
    X = _aux_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    y = np.log(sales_train.Revenue.values)
    model = LGBMRegressor(
        **LIGHTGBM_AUX_PARAMS,
        random_state=random_state,
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_lightgbm_aux(
    model: object,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
    *,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
) -> np.ndarray:
    X = _align_aux_prediction_matrix(
        val_dates,
        sales_train,
        as_of,
        feature_order,
        selected_aux_features=selected_aux_features,
    )
    return np.exp(model.predict(X))


def train_xgboost_aux(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    params: dict[str, float | int | str] | None = None,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    drop_lag_features: bool = False,
    target_mode: str = "direct",
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None = seasonal_residual_baseline,
    random_state: int = 42,
) -> tuple[object, list[str]]:
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Run `.venv\\Scripts\\python.exe -m pip install xgboost`."
        )
    X = _aux_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    X = _augment_xgboost_feature_matrix(
        X,
        sales_train.Date,
        sales_train,
        as_of,
        target_mode=target_mode,
        baseline_fn=baseline_fn,
    )
    X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
    y = np.log(sales_train.Revenue.values)
    if target_mode == "residual":
        baseline = _baseline_prediction_array(
            baseline_fn, sales_train.Date, sales_train, as_of
        )
        valid = np.isfinite(baseline)
        if not valid.any():
            raise ValueError("Residual baseline produced no valid training rows.")
        X = X.loc[valid].reset_index(drop=True)
        y = y[valid] - np.log(np.clip(baseline[valid], 1e-6, None))
    elif target_mode != "direct":
        raise ValueError(f"Unsupported XGBoost target_mode: {target_mode}")
    model_params = dict(XGBOOST_AUX_PARAMS)
    if params is not None:
        model_params.update(params)
    model = XGBRegressor(
        **model_params,
        random_state=random_state,
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_xgboost_aux(
    model: object,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
    *,
    selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    drop_lag_features: bool = False,
    target_mode: str = "direct",
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None = seasonal_residual_baseline,
) -> np.ndarray:
    X = _aux_matrix(
        val_dates,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    X = _augment_xgboost_feature_matrix(
        X,
        val_dates,
        sales_train,
        as_of,
        target_mode=target_mode,
        baseline_fn=baseline_fn,
    )
    X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_order]
    pred_log = model.predict(X)
    if target_mode == "direct":
        return np.exp(pred_log)
    if target_mode == "residual":
        baseline = _baseline_prediction_array(baseline_fn, val_dates, sales_train, as_of)
        if np.isnan(baseline).any():
            raise ValueError("Residual baseline produced invalid prediction rows.")
        return np.clip(baseline, 1e-6, None) * np.exp(pred_log)
    raise ValueError(f"Unsupported XGBoost target_mode: {target_mode}")


def train_mlp(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    random_state: int = 42,
) -> tuple[TransformedTargetRegressor, list[str]]:
    X, y = _xy(sales_train, sales_train, as_of=as_of)
    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    **MLP_PARAMS,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(200, len(X)),
            random_state=random_state,
        ),
    )
    model.fit(X, y)
    return model, list(X.columns)


def predict_mlp(
    model: TransformedTargetRegressor,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> np.ndarray:
    X = _align_prediction_matrix(
        val_dates, sales_train, as_of=as_of, feature_order=feature_order
    )
    return np.exp(model.predict(X))


def train_gbm(
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    half_life_years: float = 2.0,
    random_state: int = 42,
    use_whitelist: bool = False,
) -> tuple[HistGradientBoostingRegressor, list[str]]:
    del half_life_years, use_whitelist
    return train_hist_gbm(sales_train, as_of=as_of, random_state=random_state)


def predict_gbm(
    model: HistGradientBoostingRegressor,
    val_dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
) -> np.ndarray:
    return predict_hist_gbm(
        model, val_dates, sales_train, as_of=as_of, feature_order=feature_order
    )
