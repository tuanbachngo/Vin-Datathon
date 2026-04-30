"""Blend a feature-rich no-lag residual XGBoost model into the current baseline.

Use case
--------
The baseline model is the current safer setup:
    - XGBoost auxiliary model
    - drop_lag_features=True
    - target_mode="residual"
    - selected_aux_features=TOP_AUX_FEATURES

The feature-rich model uses the same no-lag residual target but exposes all
auxiliary trend/seasonality features (selected_aux_features=None) and applies a
more regularized XGBoost configuration. The script evaluates small blend weights
on walk-forward CV, selects a guarded best blend, then writes a Kaggle-ready
submission.

Typical command:
    python src/blend_feature_rich_model.py \
        --feature-weight-grid 0,0.05,0.10,0.15,0.20 \
        --output submissions/submission_feature_rich_blend_best.csv

Safety rules
------------
A candidate blend is auto-selected only if it:
    1. improves or matches the baseline selected score,
    2. does not worsen fold3_test_proxy RMSE,
    3. does not worsen peak-month APE beyond the tolerance.

This avoids accepting a model that looks good on average but degrades the most
future-like fold or the high-revenue months.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # package import, e.g. python -m src.blend_feature_rich_model
    from .calibration import fit_revenue_calibrator
    from .model import (
        TOP_AUX_FEATURES,
        XGBOOST_AUX_PARAMS,
        predict_xgboost_aux,
        train_xgboost_aux,
    )
    from .run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
        build_peak_month_error_reports,
        load_sales,
        load_xgboost_params,
        weighted_cv_summary,
        xgb_artifact_paths,
        xgb_runtime_config,
    )
    from .validation import default_folds, metrics
except ImportError:  # script-style import, e.g. python src/blend_feature_rich_model.py
    from calibration import fit_revenue_calibrator
    from model import (
        TOP_AUX_FEATURES,
        XGBOOST_AUX_PARAMS,
        predict_xgboost_aux,
        train_xgboost_aux,
    )
    from run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
        build_peak_month_error_reports,
        load_sales,
        load_xgboost_params,
        weighted_cv_summary,
        xgb_artifact_paths,
        xgb_runtime_config,
    )
    from validation import default_folds, metrics


MODEL_NAME_RAW = "xgboost_no_lag_residual_feature_rich_blend"
MODEL_NAME_CAL = f"{MODEL_NAME_RAW}_calibrated"
BASELINE_CONFIG_ID = "w0p000_baseline"


@dataclass(frozen=True)
class BlendResult:
    config_id: str
    feature_weight: float
    raw_weighted_rmse: float
    raw_weighted_mae: float
    raw_weighted_r2: float
    raw_weighted_mape: float
    calibrated_weighted_rmse: float
    calibrated_weighted_mae: float
    calibrated_weighted_r2: float
    calibrated_weighted_mape: float
    raw_fold3_rmse: float
    raw_fold3_mae: float
    calibrated_fold3_rmse: float
    calibrated_fold3_mae: float
    raw_peak_month_ape: float
    calibrated_peak_month_ape: float
    raw_non_peak_ape: float
    calibrated_non_peak_ape: float
    raw_peak_month_rmse: float
    calibrated_peak_month_rmse: float
    raw_non_peak_month_rmse: float
    calibrated_non_peak_month_rmse: float


def _parse_weight_grid(text: str) -> list[float]:
    weights: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Blend weight must be in [0, 1], got {value}")
        weights.append(value)
    if 0.0 not in weights:
        weights.insert(0, 0.0)
    # Keep stable order while deduplicating exact float values.
    return list(dict.fromkeys(float(w) for w in weights))


def _config_id(weight: float) -> str:
    if abs(weight) < 1e-12:
        return BASELINE_CONFIG_ID
    return f"w{str(round(weight, 4)).replace('.', 'p')}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _regularized_feature_params(depth: int) -> dict[str, float | int | str]:
    """Conservative XGBoost params for the feature-rich model.

    The goal is not to maximize standalone fit of the feature-rich model, but to
    make its errors less correlated with the baseline while reducing overfit.
    """
    params = dict(XGBOOST_AUX_PARAMS)
    params.update(
        {
            "max_depth": int(depth),
            "min_child_weight": 10.0 if int(depth) <= 3 else 12.0,
            "reg_lambda": 10.0,
            "reg_alpha": 1.0,
            "subsample": 0.80,
            "colsample_bytree": 0.65,
            "learning_rate": 0.025,
            "n_estimators": 650,
            "gamma": 0.1,
        }
    )
    return params


def _load_baseline_params() -> dict[str, float | int | str] | None:
    cfg = xgb_runtime_config(no_lag=False, no_lag_residual=True, outlier_downweight=False)
    paths = xgb_artifact_paths(str(cfg["artifact_prefix"]))
    return load_xgboost_params(paths["params"])


def _weighted_metric_rows(cv_rows: list[dict[str, Any]], folds: list) -> pd.DataFrame:
    cv = pd.DataFrame(cv_rows)
    return weighted_cv_summary(cv, folds=folds)


def _extract_fold_metric(cv_rows: list[dict[str, Any]], *, fold: str, model: str) -> dict[str, float]:
    cv = pd.DataFrame(cv_rows)
    sub = cv[(cv["fold"] == fold) & (cv["model"] == model)]
    if sub.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    row = sub.iloc[0]
    return {
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "R2": float(row["R2"]),
        "MAPE": float(row["MAPE"]),
    }


def _peak_summary_values(oof: pd.DataFrame, calibrator: Any | None) -> dict[str, float]:
    monthly, _ = build_peak_month_error_reports(
        oof[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]],
        calibrator=calibrator,
        quantile=0.75,
    )
    peak = monthly[monthly["is_peak_month"] == 1]
    non_peak = monthly[monthly["is_peak_month"] == 0]

    def _monthly_rmse(frame: pd.DataFrame, pred_col: str) -> float:
        if frame.empty:
            return float("nan")
        err = frame["actual_revenue"].to_numpy(dtype=float) - frame[pred_col].to_numpy(dtype=float)
        return float(np.sqrt(np.mean(err ** 2)))

    return {
        "raw_peak_month_ape": float(peak["ape_raw"].mean()) if len(peak) else np.nan,
        "calibrated_peak_month_ape": float(peak["ape_calibrated"].mean()) if len(peak) else np.nan,
        "raw_non_peak_ape": float(non_peak["ape_raw"].mean()) if len(non_peak) else np.nan,
        "calibrated_non_peak_ape": float(non_peak["ape_calibrated"].mean()) if len(non_peak) else np.nan,
        "raw_peak_month_rmse": _monthly_rmse(peak, "pred_raw"),
        "calibrated_peak_month_rmse": _monthly_rmse(peak, "pred_calibrated"),
        "raw_non_peak_month_rmse": _monthly_rmse(non_peak, "pred_raw"),
        "calibrated_non_peak_month_rmse": _monthly_rmse(non_peak, "pred_calibrated"),
    }


def collect_baseline_and_feature_oof(
    sales: pd.DataFrame,
    *,
    folds: list,
    baseline_params: dict[str, float | int | str] | None,
    feature_params: dict[str, float | int | str],
    random_state: int,
) -> pd.DataFrame:
    """Train baseline A and feature-rich B once per fold, return both predictions."""
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)].sort_values("Date").reset_index(drop=True)
        as_of = fold.train_end
        print(f"  fold={fold.name}, train_end={as_of.date()}, val_rows={len(val)}")

        model_a, features_a = train_xgboost_aux(
            train,
            as_of=as_of,
            params=baseline_params,
            selected_aux_features=TOP_AUX_FEATURES,
            drop_lag_features=True,
            target_mode="residual",
            outlier_downweight=False,
            random_state=random_state,
        )
        pred_a = predict_xgboost_aux(
            model_a,
            val.Date,
            sales,
            as_of,
            features_a,
            selected_aux_features=TOP_AUX_FEATURES,
            drop_lag_features=True,
            target_mode="residual",
        )

        model_b, features_b = train_xgboost_aux(
            train,
            as_of=as_of,
            params=feature_params,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
            outlier_downweight=False,
            random_state=random_state,
        )
        pred_b = predict_xgboost_aux(
            model_b,
            val.Date,
            sales,
            as_of,
            features_b,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
        )

        rows.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "train_end": as_of,
                    "Date": val.Date.to_numpy(),
                    "actual_revenue": val.Revenue.to_numpy(dtype=float),
                    "pred_baseline": pred_a,
                    "pred_feature_rich": pred_b,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_blended_oof(precomputed: pd.DataFrame, *, feature_weight: float) -> pd.DataFrame:
    frame = precomputed[["fold", "train_end", "Date", "actual_revenue"]].copy()
    w = float(np.clip(feature_weight, 0.0, 1.0))
    frame["prediction_raw"] = (
        (1.0 - w) * precomputed["pred_baseline"].to_numpy(dtype=float)
        + w * precomputed["pred_feature_rich"].to_numpy(dtype=float)
    )
    frame["config_id"] = _config_id(w)
    frame["feature_weight"] = w
    return frame


def evaluate_blend(
    oof: pd.DataFrame,
    *,
    folds: list,
    feature_weight: float,
    no_calibration: bool,
) -> tuple[BlendResult, pd.DataFrame, Any | None]:
    calibrator = None
    if not no_calibration:
        calibrator = fit_revenue_calibrator(oof).calibrator

    cv_rows: list[dict[str, Any]] = []
    for fold in folds:
        part = oof[oof["fold"] == fold.name].sort_values("Date")
        y = part["actual_revenue"].to_numpy(dtype=float)
        pred_raw = part["prediction_raw"].to_numpy(dtype=float)
        raw_m = metrics(y, pred_raw)
        cv_rows.append({"fold": fold.name, "model": MODEL_NAME_RAW, **raw_m})

        if calibrator is not None:
            pred_cal = calibrator.predict(pred_raw)
            cal_m = metrics(y, pred_cal)
            cv_rows.append({"fold": fold.name, "model": MODEL_NAME_CAL, **cal_m})

    weighted = _weighted_metric_rows(cv_rows, folds)
    raw_w = weighted[weighted["model"] == MODEL_NAME_RAW].iloc[0]
    cal_rows = weighted[weighted["model"] == MODEL_NAME_CAL]
    cal_w = cal_rows.iloc[0] if not cal_rows.empty else raw_w

    fold3_raw = _extract_fold_metric(cv_rows, fold="fold3_test_proxy", model=MODEL_NAME_RAW)
    fold3_cal = _extract_fold_metric(cv_rows, fold="fold3_test_proxy", model=MODEL_NAME_CAL)
    if np.isnan(fold3_cal["RMSE"]):
        fold3_cal = fold3_raw

    peak_vals = _peak_summary_values(oof, calibrator)

    result = BlendResult(
        config_id=_config_id(feature_weight),
        feature_weight=float(feature_weight),
        raw_weighted_rmse=float(raw_w["RMSE"]),
        raw_weighted_mae=float(raw_w["MAE"]),
        raw_weighted_r2=float(raw_w["R2"]),
        raw_weighted_mape=float(raw_w["MAPE"]),
        calibrated_weighted_rmse=float(cal_w["RMSE"]),
        calibrated_weighted_mae=float(cal_w["MAE"]),
        calibrated_weighted_r2=float(cal_w["R2"]),
        calibrated_weighted_mape=float(cal_w["MAPE"]),
        raw_fold3_rmse=float(fold3_raw["RMSE"]),
        raw_fold3_mae=float(fold3_raw["MAE"]),
        calibrated_fold3_rmse=float(fold3_cal["RMSE"]),
        calibrated_fold3_mae=float(fold3_cal["MAE"]),
        raw_peak_month_ape=float(peak_vals["raw_peak_month_ape"]),
        calibrated_peak_month_ape=float(peak_vals["calibrated_peak_month_ape"]),
        raw_non_peak_ape=float(peak_vals["raw_non_peak_ape"]),
        calibrated_non_peak_ape=float(peak_vals["calibrated_non_peak_ape"]),
        raw_peak_month_rmse=float(peak_vals["raw_peak_month_rmse"]),
        calibrated_peak_month_rmse=float(peak_vals["calibrated_peak_month_rmse"]),
        raw_non_peak_month_rmse=float(peak_vals["raw_non_peak_month_rmse"]),
        calibrated_non_peak_month_rmse=float(peak_vals["calibrated_non_peak_month_rmse"]),
    )
    cv_df = pd.DataFrame(cv_rows)
    return result, cv_df, calibrator


def select_best(
    results: pd.DataFrame,
    *,
    no_calibration: bool,
    peak_ape_tolerance_pct: float,
) -> dict[str, Any]:
    score_col = "raw_weighted_rmse" if no_calibration else "calibrated_weighted_rmse"
    fold3_col = "raw_fold3_rmse" if no_calibration else "calibrated_fold3_rmse"
    peak_col = "raw_peak_month_ape" if no_calibration else "calibrated_peak_month_ape"

    baseline = results[results["feature_weight"] == 0.0]
    if baseline.empty:
        raise RuntimeError("Baseline row feature_weight=0.0 is missing.")
    b = baseline.iloc[0]
    base_score = float(b[score_col])
    base_fold3 = float(b[fold3_col])
    base_peak = float(b[peak_col])

    df = results.copy()
    df["score_improve_pct"] = (base_score - df[score_col]) / max(base_score, 1e-12)
    df["rule_score_not_worse"] = df[score_col] <= base_score
    df["rule_fold3_not_worse"] = df[fold3_col] <= base_fold3
    df["rule_peak_not_worse"] = df[peak_col] <= base_peak * (1.0 + peak_ape_tolerance_pct / 100.0)
    df["rule_all"] = (
        df["rule_score_not_worse"]
        & df["rule_fold3_not_worse"]
        & df["rule_peak_not_worse"]
    )

    candidates = df[df["feature_weight"] > 0.0].copy()
    eligible = candidates[candidates["rule_all"]].sort_values([score_col, fold3_col])

    selected_config_id = BASELINE_CONFIG_ID
    status = "baseline_kept"
    reason = "no_candidate_passed_guards"
    manual_review_required = False

    if not eligible.empty:
        selected_config_id = str(eligible.iloc[0]["config_id"])
        status = "candidate_selected"
        reason = "candidate_passed_score_fold3_peak_guards"

    best_by_score = df.sort_values([score_col, fold3_col]).iloc[0]
    if str(best_by_score["config_id"]) != selected_config_id:
        manual_review_required = True

    selected_row = df[df["config_id"] == selected_config_id].iloc[0].to_dict()
    payload = {
        "status": status,
        "selection_reason": reason,
        "manual_review_required": bool(manual_review_required),
        "no_calibration": bool(no_calibration),
        "score_column": score_col,
        "fold3_column": fold3_col,
        "peak_column": peak_col,
        "baseline_config_id": BASELINE_CONFIG_ID,
        "baseline_score": base_score,
        "baseline_fold3_rmse": base_fold3,
        "baseline_peak_month_ape": base_peak,
        "best_by_score_config_id": str(best_by_score["config_id"]),
        "best_by_score_value": float(best_by_score[score_col]),
        "selected_config_id": selected_config_id,
        "selected_config": selected_row,
    }
    return payload, df


def predict_submission(
    sales: pd.DataFrame,
    *,
    baseline_params: dict[str, float | int | str] | None,
    feature_params: dict[str, float | int | str],
    feature_weight: float,
    calibrator: Any | None,
    random_state: int,
) -> pd.DataFrame:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    sorted_idx = sub.sort_values("Date").index.to_numpy()
    dates = sub.loc[sorted_idx, "Date"].reset_index(drop=True)
    as_of = pd.Timestamp(sales.Date.max())

    print("Training full-data baseline model A...")
    model_a, features_a = train_xgboost_aux(
        sales,
        as_of=as_of,
        params=baseline_params,
        selected_aux_features=TOP_AUX_FEATURES,
        drop_lag_features=True,
        target_mode="residual",
        outlier_downweight=False,
        random_state=random_state,
    )
    pred_a = predict_xgboost_aux(
        model_a,
        dates,
        sales,
        as_of,
        features_a,
        selected_aux_features=TOP_AUX_FEATURES,
        drop_lag_features=True,
        target_mode="residual",
    )

    w = float(np.clip(feature_weight, 0.0, 1.0))
    pred = pred_a.copy()
    if w > 0.0:
        print("Training full-data feature-rich model B...")
        model_b, features_b = train_xgboost_aux(
            sales,
            as_of=as_of,
            params=feature_params,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
            outlier_downweight=False,
            random_state=random_state,
        )
        pred_b = predict_xgboost_aux(
            model_b,
            dates,
            sales,
            as_of,
            features_b,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
        )
        pred = (1.0 - w) * pred_a + w * pred_b

    if calibrator is not None:
        pred = calibrator.predict(pred)

    out = sub.copy()
    out.loc[sorted_idx, "Revenue"] = pred
    # Preserve any other sample_submission columns, including COGS if present.
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend a regularized feature-rich no-lag residual XGBoost model into baseline."
    )
    parser.add_argument(
        "--feature-weight-grid",
        type=str,
        default="0,0.05,0.10,0.15,0.20",
        help="Comma-separated blend weights for feature-rich model B.",
    )
    parser.add_argument(
        "--feature-depth",
        type=int,
        choices=[3, 4],
        default=3,
        help="Max depth for regularized feature-rich model.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable isotonic calibration; select and submit raw blended predictions.",
    )
    parser.add_argument(
        "--peak-ape-tolerance-pct",
        type=float,
        default=1.0,
        help="Allowed peak-month APE degradation versus baseline, in percent.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission_feature_rich_blend_best.csv",
        help="Submission path. Relative paths are resolved from project ROOT.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="feature_rich_blend",
        help="Tag stored in output metadata.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for XGBoost models.",
    )
    parser.add_argument(
        "--force-best-by-score",
        action="store_true",
        help="Submit the best score config even if fold3/peak guards fail. Use with caution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = _parse_weight_grid(args.feature_weight_grid)
    sales = load_sales()
    folds = default_folds(profile="late_priority")
    baseline_params = _load_baseline_params()
    feature_params = _regularized_feature_params(args.feature_depth)

    print(
        f"Loaded sales: {len(sales)} rows, "
        f"{sales.Date.min().date()} -> {sales.Date.max().date()}"
    )
    print(f"Blend weights: {weights}")
    print(f"Feature-rich params: {json.dumps(_json_ready(feature_params), indent=2)}")

    print("\nCollecting OOF predictions for model A and model B...")
    precomputed = collect_baseline_and_feature_oof(
        sales,
        folds=folds,
        baseline_params=baseline_params,
        feature_params=feature_params,
        random_state=int(args.random_state),
    )

    result_rows: list[dict[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    oof_cache: dict[str, pd.DataFrame] = {}
    calibrator_cache: dict[str, Any | None] = {}

    for w in weights:
        cid = _config_id(w)
        print(f"\nEvaluating blend config {cid} (feature_weight={w:.4f})")
        oof = build_blended_oof(precomputed, feature_weight=w)
        result, cv_df, calibrator = evaluate_blend(
            oof,
            folds=folds,
            feature_weight=w,
            no_calibration=bool(args.no_calibration),
        )
        result_rows.append(asdict(result))
        cv_df["config_id"] = cid
        cv_df["feature_weight"] = w
        cv_frames.append(cv_df)
        oof_cache[cid] = oof
        calibrator_cache[cid] = calibrator

    raw_results = pd.DataFrame(result_rows)
    best_payload, results = select_best(
        raw_results,
        no_calibration=bool(args.no_calibration),
        peak_ape_tolerance_pct=float(args.peak_ape_tolerance_pct),
    )

    if args.force_best_by_score:
        score_col = str(best_payload["score_column"])
        forced_row = results.sort_values(score_col).iloc[0]
        best_payload["selected_config_id_before_force"] = best_payload["selected_config_id"]
        best_payload["selected_config_id"] = str(forced_row["config_id"])
        best_payload["selected_config"] = forced_row.to_dict()
        best_payload["status"] = "forced_best_by_score"
        best_payload["selection_reason"] = "--force-best-by-score was used"

    selected_id = str(best_payload["selected_config_id"])
    selected_weight = float(best_payload["selected_config"]["feature_weight"])
    selected_calibrator = calibrator_cache.get(selected_id)

    print("\nTop configs by selected score:")
    score_col = str(best_payload["score_column"])
    display_cols = [
        "config_id",
        "feature_weight",
        score_col,
        str(best_payload["fold3_column"]),
        str(best_payload["peak_column"]),
        "score_improve_pct",
        "rule_score_not_worse",
        "rule_fold3_not_worse",
        "rule_peak_not_worse",
        "rule_all",
    ]
    print(results.sort_values(score_col)[display_cols].to_string(index=False))

    print("\nSelected:")
    print(json.dumps(_json_ready(best_payload), indent=2))

    print("\nFitting full-data model(s) and writing submission...")
    submission = predict_submission(
        sales,
        baseline_params=baseline_params,
        feature_params=feature_params,
        feature_weight=selected_weight,
        calibrator=None if args.no_calibration else selected_calibrator,
        random_state=int(args.random_state),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    # Persist diagnostics.
    OUT_DIR.mkdir(exist_ok=True)
    results_path = OUT_DIR / "feature_rich_blend_results.csv"
    cv_path = OUT_DIR / "feature_rich_blend_cv_by_fold.csv"
    best_path = OUT_DIR / "feature_rich_blend_best_config.json"
    oof_path = OUT_DIR / "feature_rich_blend_best_oof.csv"
    precomputed_path = OUT_DIR / "feature_rich_blend_precomputed_oof.csv"

    results.sort_values(score_col).to_csv(results_path, index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(cv_path, index=False)
    selected_oof = oof_cache[selected_id].copy()
    if selected_calibrator is not None and not args.no_calibration:
        selected_oof["prediction_calibrated"] = selected_calibrator.predict(
            selected_oof["prediction_raw"].to_numpy()
        )
    else:
        selected_oof["prediction_calibrated"] = selected_oof["prediction_raw"]
    selected_oof.to_csv(oof_path, index=False)
    precomputed.to_csv(precomputed_path, index=False)

    metadata = {
        "tag": args.tag,
        "output_submission": str(output_path),
        "feature_weight_grid": weights,
        "feature_depth": int(args.feature_depth),
        "feature_rich_params": feature_params,
        "baseline_params_source": "xgboost_no_lag_residual latest saved json if present, else defaults",
        **best_payload,
    }
    with best_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metadata), handle, indent=2, ensure_ascii=True)

    print(f"\nWrote submission: {output_path}")
    print(f"Wrote diagnostics: {results_path}")
    print(f"Wrote diagnostics: {cv_path}")
    print(f"Wrote diagnostics: {best_path}")
    print(f"Wrote diagnostics: {oof_path}")
    print(f"Wrote diagnostics: {precomputed_path}")


if __name__ == "__main__":
    main()
