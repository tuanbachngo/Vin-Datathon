from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak_refined6")
    if not bak.exists():
        shutil.copy2(path, bak)


def find_function_bounds(text: str, name: str) -> tuple[int, int]:
    start = text.find(f"\ndef {name}(")
    if start == -1:
        if text.startswith(f"def {name}("):
            start = 0
        else:
            raise ValueError(f"Function {name} not found")
    else:
        start += 1
    candidates = []
    for marker in ("\ndef ", "\nclass "):
        idx = text.find(marker, start + 1)
        if idx != -1:
            candidates.append(idx + 1)
    return start, min(candidates) if candidates else len(text)


def replace_function(text: str, name: str, new_code: str) -> str:
    start, end = find_function_bounds(text, name)
    return text[:start] + new_code.rstrip() + "\n\n" + text[end:]


def insert_before_function(text: str, name: str, block: str) -> str:
    start, _ = find_function_bounds(text, name)
    return text[:start] + block.rstrip() + "\n\n" + text[start:]


def patch_baselines(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    if "def _safe_divide(" not in text:
        safe_divide = '''
def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > 1e-9))
    return out
'''
        _, end_safe_mean = find_function_bounds(text, "_safe_mean")
        text = text[:end_safe_mean].rstrip() + safe_divide + text[end_safe_mean:]

    if "def seasonal_lookup_level_adjusted(" not in text:
        if "def _seasonal_lookup_level_adjusted(" not in text:
            raise ValueError("Need _seasonal_lookup_level_adjusted or seasonal_lookup_level_adjusted in baselines.py")
        text = text.replace(
            "def _seasonal_lookup_level_adjusted(",
            "def seasonal_lookup_level_adjusted(",
            1,
        )
        _, end = find_function_bounds(text, "seasonal_lookup_level_adjusted")
        alias = "\n\n_seasonal_lookup_level_adjusted = seasonal_lookup_level_adjusted\n"
        text = text[:end].rstrip() + alias + text[end:]

    if "def seasonal_residual_baseline_components(" not in text:
        components = '''
def seasonal_residual_baseline_components(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    mean_2y = np.asarray(seasonal_naive_mean_2y(target_dates, sales, as_of), dtype=float)
    growth = np.asarray(seasonal_naive_growth_adjusted(target_dates, sales, as_of), dtype=float)
    lookup = np.asarray(seasonal_lookup_level_adjusted(target_dates, sales, as_of), dtype=float)

    anchor = np.asarray(mean_2y, dtype=float)
    anchor = np.where(np.isfinite(anchor), anchor, lookup)
    anchor = np.where(np.isfinite(anchor) & (anchor > 0), anchor, np.nan)
    if np.isnan(anchor).any():
        safe_fill = np.nanmedian(lookup) if np.isfinite(lookup).any() else 1.0
        anchor = np.where(np.isnan(anchor), safe_fill, anchor)

    consensus = _safe_mean(mean_2y, growth, lookup)
    consensus = np.where(np.isfinite(consensus) & (consensus > 0), consensus, anchor)

    component_stack = np.vstack([mean_2y, growth, lookup]).astype(float)
    component_max = np.nanmax(component_stack, axis=0)
    component_min = np.nanmin(component_stack, axis=0)
    spread = _safe_divide(component_max - component_min, consensus)

    df = pd.DataFrame({
        "baseline_mean2y": mean_2y,
        "baseline_growth": growth,
        "baseline_lookup": lookup,
        "baseline_anchor": anchor,
        "baseline_consensus": consensus,
        "baseline_spread": spread,
    })
    for col in [
        "baseline_mean2y",
        "baseline_growth",
        "baseline_lookup",
        "baseline_anchor",
        "baseline_consensus",
    ]:
        df[f"log_{col}"] = np.log(np.maximum(df[col].astype(float), 1.0))

    pairs = [
        ("mean2y", mean_2y, "lookup", lookup),
        ("growth", growth, "lookup", lookup),
        ("mean2y", mean_2y, "growth", growth),
        ("anchor", anchor, "consensus", consensus),
    ]
    for left_name, left, right_name, right in pairs:
        df[f"gap_{left_name}_vs_{right_name}"] = left - right
        df[f"ratio_{left_name}_vs_{right_name}"] = _safe_divide(left, right)

    for col in [c for c in df.columns if c.startswith("ratio_")]:
        df[col] = df[col].clip(lower=0.1, upper=10.0)

    return df.replace([np.inf, -np.inf], np.nan)
'''
        text = insert_before_function(text, "seasonal_residual_baseline", components)

    residual_baseline = '''
def seasonal_residual_baseline(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    return seasonal_residual_baseline_components(
        target_dates, sales, as_of
    )["baseline_anchor"].to_numpy(dtype=float)
'''
    text = replace_function(text, "seasonal_residual_baseline", residual_baseline)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(f"[OK] patched {path}")
    else:
        print(f"[SKIP] no changes needed for {path}")


def patch_aux_features(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    if "def _safe_ratio_feature(" not in text:
        safe_ratio = '''
def _safe_ratio_feature(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    neutral: float = 1.0,
    lower: float = 0.1,
    upper: float = 5.0,
) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(num.shape, neutral, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > 1e-9))
    out = np.where(np.isfinite(out), out, neutral)
    return np.clip(out, lower, upper)
'''
        text = insert_before_function(text, "_project_trend_seasonal", safe_ratio)

    project_func = '''
def _project_trend_seasonal(
    model: dict | None,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    n = len(target_dates)
    columns = [
        "trend",
        "trend_month",
        "trend_month_dow",
        "month_lift",
        "dow_lift",
        "month_ratio",
        "dow_ratio",
    ]
    if model is None:
        return pd.DataFrame({c: np.full(n, np.nan) for c in columns})

    days = (target_dates - model["origin"]).days.values.astype(float)
    trend = model["slope"] * days + model["intercept"]

    month_adj = np.array([
        model["month_seasonal"].get(m, 0.0) for m in target_dates.month
    ])
    dow_adj = np.array([
        model["dow_seasonal"].get(d, 0.0) for d in target_dates.dayofweek
    ])

    trend_month = trend + month_adj
    trend_month_dow = trend_month + dow_adj
    month_lift = trend_month - trend
    dow_lift = trend_month_dow - trend_month

    return pd.DataFrame({
        "trend": trend,
        "trend_month": trend_month,
        "trend_month_dow": trend_month_dow,
        "month_lift": month_lift,
        "dow_lift": dow_lift,
        "month_ratio": _safe_ratio_feature(trend_month, trend),
        "dow_ratio": _safe_ratio_feature(trend_month_dow, trend_month),
    })
'''
    text = replace_function(text, "_project_trend_seasonal", project_func)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(f"[OK] patched {path}")
    else:
        print(f"[SKIP] no changes needed for {path}")


def patch_model(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    text = text.replace(
        "from .baselines import seasonal_residual_baseline",
        "from .baselines import seasonal_residual_baseline, seasonal_residual_baseline_components",
    )
    text = text.replace(
        "from baselines import seasonal_residual_baseline",
        "from baselines import seasonal_residual_baseline, seasonal_residual_baseline_components",
    )

    if "XGB_AUX_NON_LAG_SUFFIXES" in text and "XGB_LAG_LIKE_TOKENS" not in text:
        start = text.index("XGB_AUX_NON_LAG_SUFFIXES = (")
        end = text.index("\n\nMLP_PARAMS", start)
        const_block = '''# Tokens used by no-lag XGBoost mode. Drops only genuine lag-like columns.
XGB_LAG_LIKE_TOKENS = (
    "rev_lag_",
    "log_rev_lag_",
    "_lag",
    "lag_",
    "_year1",
    "_year2",
    "_roll",
    "roll",
    "_ewm",
    "ewm",
    "_365",
    "_548",
    "_730",
    "_1095",
    "end_365d_ago",
    "end_548d_ago",
    "end_730d_ago",
    "end_1095d_ago",
)
'''
        text = text[:start] + const_block + text[end:]

    helper_block = '''
def _is_lag_like_feature(column: str) -> bool:
    c = column.lower()
    return any(token in c for token in XGB_LAG_LIKE_TOKENS)


def _filter_xgboost_feature_matrix(
    X: pd.DataFrame,
    *,
    drop_lag_features: bool,
) -> pd.DataFrame:
    if not drop_lag_features:
        return X
    keep_columns = [col for col in X.columns if not _is_lag_like_feature(col)]
    return X[keep_columns]


def _residual_baseline_feature_matrix(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    use_default = baseline_fn is None or baseline_fn is seasonal_residual_baseline
    if use_default:
        return seasonal_residual_baseline_components(dates, sales, as_of).reset_index(drop=True)

    anchor = _baseline_prediction_array(baseline_fn, dates, sales, as_of)
    out = pd.DataFrame({"baseline_anchor": anchor})
    out["log_baseline_anchor"] = np.log(np.maximum(out["baseline_anchor"], 1.0))
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def _add_residual_baseline_features(
    X: pd.DataFrame,
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    baseline_features = _residual_baseline_feature_matrix(
        dates, sales, as_of, baseline_fn
    )
    return pd.concat(
        [X.reset_index(drop=True), baseline_features.reset_index(drop=True)],
        axis=1,
    )


def _build_xgboost_aux_matrix(
    dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    selected_aux_features: list[str] | None,
    drop_lag_features: bool,
    target_mode: str,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    X = _aux_matrix(
        dates,
        sales,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
    if target_mode == "residual":
        X = _add_residual_baseline_features(X, dates, sales, as_of, baseline_fn)
    return X


def _align_xgboost_aux_prediction_matrix(
    dates: pd.Series,
    sales_train: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_order: list[str],
    *,
    selected_aux_features: list[str] | None,
    drop_lag_features: bool,
    target_mode: str,
    baseline_fn: Callable[[pd.Series, pd.DataFrame, pd.Timestamp], np.ndarray] | None,
) -> pd.DataFrame:
    X = _build_xgboost_aux_matrix(
        dates,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
        drop_lag_features=drop_lag_features,
        target_mode=target_mode,
        baseline_fn=baseline_fn,
    )
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_order]
'''
    if "def _is_lag_like_feature(" not in text:
        text = replace_function(text, "_filter_xgboost_feature_matrix", helper_block)

    old_train = '''    X = _aux_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
    )
    X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
    y = np.log(sales_train.Revenue.values)
'''
    new_train = '''    X = _build_xgboost_aux_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        selected_aux_features=selected_aux_features,
        drop_lag_features=drop_lag_features,
        target_mode=target_mode,
        baseline_fn=baseline_fn,
    )
    y = np.log(sales_train.Revenue.values)
'''
    if old_train in text:
        text = text.replace(old_train, new_train)

    old_predict = '''    X = _align_aux_prediction_matrix(
        val_dates,
        sales_train,
        as_of,
        feature_order,
        selected_aux_features=selected_aux_features,
    )
    X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
    pred_log = model.predict(X)
'''
    new_predict = '''    X = _align_xgboost_aux_prediction_matrix(
        val_dates,
        sales_train,
        as_of,
        feature_order,
        selected_aux_features=selected_aux_features,
        drop_lag_features=drop_lag_features,
        target_mode=target_mode,
        baseline_fn=baseline_fn,
    )
    pred_log = model.predict(X)
'''
    if old_predict in text:
        text = text.replace(old_predict, new_predict)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(f"[OK] patched {path}")
    else:
        print(f"[SKIP] no changes needed for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply refined6 residual/no-lag feature patch.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing src/")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    src = root / "src"
    files = {
        "baselines": src / "baselines.py",
        "aux_features": src / "aux_features.py",
        "model": src / "model.py",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)

    patch_baselines(files["baselines"])
    patch_aux_features(files["aux_features"])
    patch_model(files["model"])

    print("\nDone. Backup files were written as *.bak_refined6 on first patch.")
    print("Suggested smoke test:")
    print('  python src/tune_feature_rich_regularization.py --max-configs 2 --output "submissions/submission_refined6_smoke.csv"')


if __name__ == "__main__":
    main()
