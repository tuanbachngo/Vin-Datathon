"""MIC-style auxiliary feature ranking for XGBoost pipelines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from aux_features import aux_feature_groups, build_aux_daily
from features import LOOKUP_HISTORY_MODE_RECENT_REGIME
from model import (
    TOP_AUX_FEATURES,
    _aux_matrix,
    _baseline_prediction_array,
    _filter_xgboost_feature_matrix,
)
from validation import Fold, default_folds

try:  # pragma: no cover - optional dependency
    from minepy import MINE
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    MINE = None


MIC_REGIME_CUTOFF = pd.Timestamp("2020-01-01")
MIC_FAMILY_SUFFIXES = (
    "_trend_month_dow",
    "_trend_month",
    "_trend",
)
MIC_SUBSETS = ("all", "pre2020", "post2020")


@dataclass
class MICFeatureSelectionResult:
    selected_features: list[str]
    feature_scores: pd.DataFrame
    group_summary: pd.DataFrame
    summary: dict[str, object]


def _mic_backend_name() -> str:
    return "minepy" if MINE is not None else "quantile_mi_approx"


def _quantile_codes(values: np.ndarray, bins: int) -> np.ndarray | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, copy=False)
    if series.nunique(dropna=True) < 2:
        return None
    try:
        codes = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    if codes is None:
        return None
    out = np.asarray(codes, dtype=float)
    valid = np.isfinite(out)
    if not valid.any():
        return None
    unique_codes = np.unique(out[valid].astype(int))
    if len(unique_codes) < 2:
        return None
    return out.astype(int)


def _approximate_mic_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_bins: int,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < 25 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    upper_bins = min(max_bins, max(2, int(np.sqrt(len(x)))))
    y_codes_cache: dict[int, tuple[np.ndarray, int]] = {}
    for bins in range(2, upper_bins + 1):
        codes = _quantile_codes(y, bins)
        if codes is None:
            continue
        y_codes_cache[bins] = (codes, len(np.unique(codes)))

    best = 0.0
    for x_bins in range(2, upper_bins + 1):
        x_codes = _quantile_codes(x, x_bins)
        if x_codes is None:
            continue
        x_cardinality = len(np.unique(x_codes))
        for y_codes, y_cardinality in y_codes_cache.values():
            denom = np.log(min(x_cardinality, y_cardinality))
            if denom <= 0:
                continue
            score = mutual_info_score(x_codes, y_codes) / denom
            best = max(best, float(score))
    return float(np.clip(best, 0.0, 1.0))


def _minepy_mic_score(x: np.ndarray, y: np.ndarray) -> float:
    if MINE is None:  # pragma: no cover - guarded by caller
        return float("nan")
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < 25 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    mine = MINE()
    mine.compute_score(x, y)
    return float(mine.mic())


def _mic_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_bins: int,
) -> float:
    if MINE is not None:
        return _minepy_mic_score(x, y)
    return _approximate_mic_score(x, y, max_bins=max_bins)


def _split_aux_feature_name(feature: str) -> tuple[str, str]:
    if not feature.startswith("aux_"):
        return feature, "other"
    body = feature[4:]
    for suffix in sorted(MIC_FAMILY_SUFFIXES, key=len, reverse=True):
        if body.endswith(suffix):
            return body[: -len(suffix)], suffix.removeprefix("_")
    return body, "other"


def _raw_aux_group_map() -> dict[str, str]:
    aux_daily = build_aux_daily()
    groups = aux_feature_groups(aux_daily)
    mapping: dict[str, str] = {}
    for group_name, columns in groups.items():
        if group_name == "all_aux":
            continue
        for column in columns:
            mapping[column] = group_name
    return mapping


def _score_subset(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    fold_name: str,
    subset_name: str,
    max_bins: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in X.columns:
        score = _mic_score(X[feature].to_numpy(dtype=float), y, max_bins=max_bins)
        rows.append(
            {
                "fold": fold_name,
                "subset": subset_name,
                "feature": feature,
                "mic_score": score,
            }
        )
    return rows


def select_aux_features_via_mic(
    sales: pd.DataFrame,
    *,
    folds: list[Fold] | None = None,
    drop_lag_features: bool = True,
    target_mode: str = "residual",
    lookup_history_mode: str = LOOKUP_HISTORY_MODE_RECENT_REGIME,
    top_n: int = len(TOP_AUX_FEATURES),
    family_top_k: int = 2,
    stability_threshold: float = 0.15,
    min_regime_rows: int = 180,
    max_bins: int = 8,
) -> MICFeatureSelectionResult:
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}.")
    if family_top_k <= 0:
        raise ValueError(f"family_top_k must be positive, got {family_top_k}.")
    if max_bins < 2:
        raise ValueError(f"max_bins must be at least 2, got {max_bins}.")

    folds = default_folds() if folds is None else folds
    score_rows: list[dict[str, object]] = []

    for fold in folds:
        train = sales[sales.Date <= fold.train_end].reset_index(drop=True)
        X = _aux_matrix(
            train.Date,
            train,
            fold.train_end,
            selected_aux_features=None,
            lookup_history_mode=lookup_history_mode,
        )
        X = _filter_xgboost_feature_matrix(X, drop_lag_features=drop_lag_features)
        aux_columns = [column for column in X.columns if column.startswith("aux_")]
        if not aux_columns:
            raise ValueError("No auxiliary features available for MIC scoring.")

        y = np.log(train.Revenue.to_numpy(dtype=float))
        train_dates = pd.to_datetime(train.Date).reset_index(drop=True)
        if target_mode == "residual":
            baseline = _baseline_prediction_array(
                None,
                train.Date,
                train,
                fold.train_end,
                lookup_history_mode=lookup_history_mode,
            )
            valid = np.isfinite(baseline)
            if not valid.any():
                raise ValueError("Residual baseline produced no valid rows for MIC.")
            X = X.loc[valid, aux_columns].reset_index(drop=True)
            y = y[valid] - np.log(np.clip(baseline[valid], 1e-6, None))
            train_dates = train_dates.loc[valid].reset_index(drop=True)
        elif target_mode == "direct":
            X = X[aux_columns].reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        score_rows.extend(
            _score_subset(
                X,
                y,
                fold_name=fold.name,
                subset_name="all",
                max_bins=max_bins,
            )
        )

        pre_mask = train_dates < MIC_REGIME_CUTOFF
        if int(pre_mask.sum()) >= min_regime_rows:
            score_rows.extend(
                _score_subset(
                    X.loc[pre_mask].reset_index(drop=True),
                    y[pre_mask.to_numpy()],
                    fold_name=fold.name,
                    subset_name="pre2020",
                    max_bins=max_bins,
                )
            )

        post_mask = ~pre_mask
        if int(post_mask.sum()) >= min_regime_rows:
            score_rows.extend(
                _score_subset(
                    X.loc[post_mask].reset_index(drop=True),
                    y[post_mask.to_numpy()],
                    fold_name=fold.name,
                    subset_name="post2020",
                    max_bins=max_bins,
                )
            )

    score_frame = pd.DataFrame(score_rows)
    if score_frame.empty:
        raise ValueError("MIC feature scoring produced no rows.")

    score_frame["rank_within_subset"] = score_frame.groupby(
        ["fold", "subset"]
    )["mic_score"].rank(ascending=False, method="dense")

    score_pivot = (
        score_frame.pivot_table(
            index="feature",
            columns="subset",
            values="mic_score",
            aggfunc="median",
        )
        .rename(
            columns={
                "all": "median_mic_all",
                "pre2020": "median_mic_pre2020",
                "post2020": "median_mic_post2020",
            }
        )
    )
    for column in (
        "median_mic_all",
        "median_mic_pre2020",
        "median_mic_post2020",
    ):
        if column not in score_pivot.columns:
            score_pivot[column] = np.nan
    rank_pivot = (
        score_frame.pivot_table(
            index="feature",
            columns="subset",
            values="rank_within_subset",
            aggfunc="median",
        )
        .rename(
            columns={
                "all": "median_rank_all",
                "pre2020": "median_rank_pre2020",
                "post2020": "median_rank_post2020",
            }
        )
    )
    for column in (
        "median_rank_all",
        "median_rank_pre2020",
        "median_rank_post2020",
    ):
        if column not in rank_pivot.columns:
            rank_pivot[column] = np.nan
    fold_coverage = (
        score_frame[score_frame["subset"] == "all"]
        .groupby("feature")
        .size()
        .rename("fold_coverage_all")
    )

    feature_scores = pd.concat([score_pivot, rank_pivot, fold_coverage], axis=1)
    feature_scores = feature_scores.sort_values(
        by=["median_mic_all", "median_rank_all"],
        ascending=[False, True],
    )

    raw_group_map = _raw_aux_group_map()
    feature_scores["raw_aux_feature"] = [
        _split_aux_feature_name(feature)[0] for feature in feature_scores.index
    ]
    feature_scores["family_variant"] = [
        _split_aux_feature_name(feature)[1] for feature in feature_scores.index
    ]
    feature_scores["source_group"] = feature_scores["raw_aux_feature"].map(raw_group_map)

    pre_scores = feature_scores["median_mic_pre2020"]
    post_scores = feature_scores["median_mic_post2020"]
    numerator = np.minimum(pre_scores, post_scores)
    denominator = np.maximum(pre_scores, post_scores)
    feature_scores["stability_ratio"] = np.where(
        np.isfinite(pre_scores) & np.isfinite(post_scores) & (denominator > 0),
        numerator / denominator,
        np.nan,
    )
    feature_scores["passes_stability"] = (
        feature_scores["stability_ratio"].isna()
        | (feature_scores["stability_ratio"] >= stability_threshold)
    )

    feature_scores["family_rank"] = feature_scores.groupby("raw_aux_feature")[
        "median_mic_all"
    ].rank(ascending=False, method="first")

    shortlisted = feature_scores[
        feature_scores["passes_stability"] & (feature_scores["family_rank"] <= family_top_k)
    ].copy()
    if shortlisted.empty:
        shortlisted = feature_scores[
            feature_scores["family_rank"] <= family_top_k
        ].copy()
    shortlisted = shortlisted.sort_values(
        by=["median_mic_all", "median_rank_all"],
        ascending=[False, True],
    )
    selected_features = shortlisted.head(top_n).index.tolist()
    feature_scores["selected"] = feature_scores.index.isin(selected_features)

    group_summary = (
        feature_scores.groupby("source_group", dropna=False)
        .agg(
            feature_count=("median_mic_all", "size"),
            selected_count=("selected", "sum"),
            median_mic_all=("median_mic_all", "median"),
            max_mic_all=("median_mic_all", "max"),
        )
        .reset_index()
        .sort_values(by=["selected_count", "median_mic_all"], ascending=[False, False])
    )

    summary: dict[str, object] = {
        "backend": _mic_backend_name(),
        "candidate_feature_count": int(len(feature_scores)),
        "selected_feature_count": int(len(selected_features)),
        "top_n": int(top_n),
        "family_top_k": int(family_top_k),
        "stability_threshold": float(stability_threshold),
        "min_regime_rows": int(min_regime_rows),
        "max_bins": int(max_bins),
        "fold_count": int(len(folds)),
        "lookup_history_mode": lookup_history_mode,
        "target_mode": target_mode,
        "drop_lag_features": bool(drop_lag_features),
        "selected_features": selected_features,
    }

    return MICFeatureSelectionResult(
        selected_features=selected_features,
        feature_scores=feature_scores.reset_index().rename(columns={"index": "feature"}),
        group_summary=group_summary,
        summary=summary,
    )
