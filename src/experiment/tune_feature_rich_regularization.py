"""Tune regularization around the refined5 feature-rich XGBoost setup.

Baseline v2 / refined5:
- Feature-rich XGBoost
- drop_lag_features=True
- target_mode="residual"
- selected_aux_features=None
- isotonic calibration enabled by default

This script does NOT run the full run_pipeline model zoo. It only evaluates curated
feature-rich XGBoost regularization configs, chooses the best one using late-priority
walk-forward CV guards, and writes one submission file.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # package import: python -m src.tune_feature_rich_regularization
    from .model import predict_xgboost_aux, train_xgboost_aux
    from .run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
        build_peak_month_error_reports,
        default_folds,
        fit_revenue_calibrator,
        load_sales,
        metrics,
    )
except ImportError:  # script-style import: python src/tune_feature_rich_regularization.py
    from model import predict_xgboost_aux, train_xgboost_aux
    from run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
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


@dataclass(frozen=True)
class RegularizationConfig:
    config_id: str
    description: str
    params: dict[str, float | int | str]


def build_candidate_configs() -> list[RegularizationConfig]:
    """Small curated search around refined5; no large grid."""
    specs: list[tuple[str, str, dict[str, float | int | str]]] = [
        (
            "refined5_baseline",
            "Current best refined5 params",
            {},
        ),
        (
            "depth2_same_reg",
            "Shallower trees, same regularization",
            {"max_depth": 2, "min_child_weight": 10.0, "reg_lambda": 10.0},
        ),
        (
            "depth3_more_child_more_l2",
            "More conservative child weight and L2",
            {"max_depth": 3, "min_child_weight": 14.0, "reg_lambda": 15.0},
        ),
        (
            "depth3_less_child_less_l2",
            "Slightly less conservative around refined5",
            {"max_depth": 3, "min_child_weight": 8.0, "reg_lambda": 8.0},
        ),
        (
            "depth4_strong_reg",
            "Deeper trees but stronger regularization",
            {"max_depth": 4, "min_child_weight": 14.0, "reg_lambda": 20.0},
        ),
        (
            "low_colsample_more_l2",
            "Lower column sample, stronger L2",
            {"colsample_bytree": 0.55, "reg_lambda": 15.0},
        ),
        (
            "higher_colsample_same_l2",
            "Higher column sample, same L2",
            {"colsample_bytree": 0.75, "reg_lambda": 10.0},
        ),
        (
            "more_l1_more_l2",
            "More L1 and L2 regularization",
            {"reg_alpha": 2.0, "reg_lambda": 15.0},
        ),
        (
            "less_l1_less_l2",
            "Less L1 and L2 regularization",
            {"reg_alpha": 0.5, "reg_lambda": 8.0},
        ),
        (
            "slower_lr_more_trees",
            "Lower learning rate and more trees",
            {"learning_rate": 0.02, "n_estimators": 800},
        ),
        (
            "faster_lr_fewer_trees",
            "Higher learning rate and fewer trees",
            {"learning_rate": 0.03, "n_estimators": 550},
        ),
        (
            "depth2_stronger_reg",
            "Very conservative depth-2 model",
            {
                "max_depth": 2,
                "min_child_weight": 14.0,
                "reg_alpha": 2.0,
                "reg_lambda": 20.0,
                "colsample_bytree": 0.60,
            },
        ),
    ]

    configs: list[RegularizationConfig] = []
    for config_id, desc, overrides in specs:
        params = dict(REFINED5_PARAMS)
        params.update(overrides)
        configs.append(RegularizationConfig(config_id, desc, params))
    return configs


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _weighted_average(rows: list[dict[str, Any]], folds: list, *, prefix: str) -> dict[str, float]:
    fold_weight = {f.name: float(getattr(f, "weight", 1.0)) for f in folds}
    weights = np.asarray([fold_weight.get(str(r["fold"]), 1.0) for r in rows], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(rows), dtype=float)

    out: dict[str, float] = {}
    for metric_name in ["MAE", "RMSE", "R2", "MAPE"]:
        vals = np.asarray([float(r[metric_name]) for r in rows], dtype=float)
        out[f"{prefix}_weighted_{metric_name.lower()}"] = float(np.average(vals, weights=weights))
    return out


def collect_feature_rich_oof(
    sales: pd.DataFrame,
    *,
    folds: list,
    params: dict[str, float | int | str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Collect OOF predictions for one feature-rich no-lag residual config."""
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end].copy()
        val = sales[fold.mask_val(sales.Date)].sort_values("Date").reset_index(drop=True)
        as_of = fold.train_end

        model, feature_order = train_xgboost_aux(
            train,
            as_of=as_of,
            params=params,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
            outlier_downweight=False,
            random_state=random_state,
        )
        pred = predict_xgboost_aux(
            model,
            val.Date,
            sales,
            as_of,
            feature_order,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode="residual",
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


def evaluate_oof(
    oof: pd.DataFrame,
    *,
    folds: list,
    no_calibration: bool,
    peak_month_quantile: float,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    """Return summary row, fold-level metrics, and fitted calibrator if enabled."""
    calibrator = None
    frame = oof.copy()
    frame["prediction_calibrated"] = frame["prediction_raw"].astype(float)

    if not no_calibration:
        cal_result = fit_revenue_calibrator(
            frame[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]]
        )
        calibrator = cal_result.calibrator
        frame["prediction_calibrated"] = calibrator.predict(frame["prediction_raw"].to_numpy())

    fold_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []

    for fold_name, part in frame.groupby("fold", sort=False):
        raw_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_raw"].to_numpy())
        cal_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_calibrated"].to_numpy())

        raw_row = {"fold": fold_name, "prediction_type": "raw", **raw_m}
        cal_row = {"fold": fold_name, "prediction_type": "calibrated", **cal_m}
        fold_rows.extend([raw_row, cal_row])
        raw_rows.append({"fold": fold_name, **raw_m})
        cal_rows.append({"fold": fold_name, **cal_m})

    raw_weighted = _weighted_average(raw_rows, folds, prefix="raw")
    cal_weighted = _weighted_average(cal_rows, folds, prefix="calibrated")

    fold3_raw = next((r for r in raw_rows if r["fold"] == "fold3_test_proxy"), None)
    fold3_cal = next((r for r in cal_rows if r["fold"] == "fold3_test_proxy"), None)
    if fold3_raw is None or fold3_cal is None:
        raise RuntimeError("fold3_test_proxy not found in fold metrics")

    peak_monthly, peak_summary = build_peak_month_error_reports(
        frame[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]],
        calibrator=calibrator,
        quantile=float(peak_month_quantile),
    )
    del peak_monthly

    peak_all = peak_summary[
        (peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "peak_month")
    ]
    non_peak_all = peak_summary[
        (peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "non_peak")
    ]

    peak_row = peak_all.iloc[0] if not peak_all.empty else None
    non_peak_row = non_peak_all.iloc[0] if not non_peak_all.empty else None

    summary = {
        **raw_weighted,
        **cal_weighted,
        "raw_fold3_rmse": float(fold3_raw["RMSE"]),
        "raw_fold3_mae": float(fold3_raw["MAE"]),
        "calibrated_fold3_rmse": float(fold3_cal["RMSE"]),
        "calibrated_fold3_mae": float(fold3_cal["MAE"]),
        "raw_peak_month_ape": (
            float(peak_row["avg_monthly_ape_raw"]) if peak_row is not None else np.nan
        ),
        "calibrated_peak_month_ape": (
            float(peak_row["avg_monthly_ape_calibrated"]) if peak_row is not None else np.nan
        ),
        "raw_non_peak_ape": (
            float(non_peak_row["avg_monthly_ape_raw"]) if non_peak_row is not None else np.nan
        ),
        "calibrated_non_peak_ape": (
            float(non_peak_row["avg_monthly_ape_calibrated"]) if non_peak_row is not None else np.nan
        ),
        "raw_peak_month_rmse": (
            float(peak_row["avg_monthly_abs_err_raw"]) if peak_row is not None else np.nan
        ),
        "calibrated_peak_month_rmse": (
            float(peak_row["avg_monthly_abs_err_calibrated"]) if peak_row is not None else np.nan
        ),
        "raw_non_peak_month_rmse": (
            float(non_peak_row["avg_monthly_abs_err_raw"]) if non_peak_row is not None else np.nan
        ),
        "calibrated_non_peak_month_rmse": (
            float(non_peak_row["avg_monthly_abs_err_calibrated"])
            if non_peak_row is not None
            else np.nan
        ),
    }
    return summary, pd.DataFrame(fold_rows), calibrator


def select_best(
    results: pd.DataFrame,
    *,
    no_calibration: bool,
    force_best_by_score: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = results.copy()
    base = df[df["config_id"] == "refined5_baseline"]
    if base.empty:
        raise RuntimeError("refined5_baseline row missing")
    b = base.iloc[0]

    score_col = "raw_weighted_rmse" if no_calibration else "calibrated_weighted_rmse"
    fold3_col = "raw_fold3_rmse" if no_calibration else "calibrated_fold3_rmse"
    peak_col = "raw_peak_month_ape" if no_calibration else "calibrated_peak_month_ape"

    baseline_score = float(b[score_col])
    baseline_fold3 = float(b[fold3_col])
    baseline_peak = float(b[peak_col])

    df["score_improve_pct"] = (baseline_score - df[score_col]) / max(baseline_score, 1e-12)
    df["rule_score_not_worse"] = df[score_col] <= baseline_score
    df["rule_fold3_not_worse"] = df[fold3_col] <= baseline_fold3
    df["rule_peak_not_worse"] = df[peak_col] <= baseline_peak
    df["rule_all"] = (
        df["rule_score_not_worse"] & df["rule_fold3_not_worse"] & df["rule_peak_not_worse"]
    )

    ranked = df.sort_values([score_col, fold3_col], ascending=[True, True]).reset_index(drop=True)
    best_by_score = ranked.iloc[0]
    eligible = ranked[ranked["rule_all"] & (ranked["config_id"] != "refined5_baseline")]

    selected_id = "refined5_baseline"
    status = "baseline_kept"
    reason = "no_candidate_passed_guards"

    if force_best_by_score:
        selected_id = str(best_by_score["config_id"])
        status = "forced_best_by_score"
        reason = "force_best_by_score_enabled"
    elif not eligible.empty:
        selected_id = str(eligible.iloc[0]["config_id"])
        status = "candidate_selected"
        reason = "candidate_passed_score_fold3_peak_guards"

    selected_row = df[df["config_id"] == selected_id].iloc[0].to_dict()
    payload = {
        "status": status,
        "selection_reason": reason,
        "manual_review_required": bool(force_best_by_score and not bool(best_by_score["rule_all"])),
        "no_calibration": bool(no_calibration),
        "score_column": score_col,
        "fold3_column": fold3_col,
        "peak_column": peak_col,
        "baseline_config_id": "refined5_baseline",
        "baseline_score": baseline_score,
        "baseline_fold3_rmse": baseline_fold3,
        "baseline_peak_month_ape": baseline_peak,
        "best_by_score_config_id": str(best_by_score["config_id"]),
        "best_by_score_value": float(best_by_score[score_col]),
        "selected_config_id": selected_id,
        "selected_config": selected_row,
    }
    return payload, df


def predict_submission(
    sales: pd.DataFrame,
    *,
    params: dict[str, float | int | str],
    calibrator: Any,
    no_calibration: bool,
    output: Path,
    random_state: int = 42,
) -> Path:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = pd.Timestamp(sales["Date"].max())

    model, feature_order = train_xgboost_aux(
        sales,
        as_of=as_of,
        params=params,
        selected_aux_features=None,
        drop_lag_features=True,
        target_mode="residual",
        outlier_downweight=False,
        random_state=random_state,
    )
    pred_raw = predict_xgboost_aux(
        model,
        sub.Date,
        sales,
        as_of,
        feature_order,
        selected_aux_features=None,
        drop_lag_features=True,
        target_mode="residual",
    )
    pred = pred_raw if no_calibration or calibrator is None else calibrator.predict(pred_raw)

    out = sub.copy()
    out["Revenue"] = pred
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curated regularization tuning around refined5 feature-rich XGBoost."
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Limit evaluated configs for quick tests. 0 = all curated configs.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable isotonic calibration and select by raw metrics.",
    )
    parser.add_argument(
        "--force-best-by-score",
        action="store_true",
        help="Select the best weighted RMSE config even if fold3/peak guards fail.",
    )
    parser.add_argument(
        "--peak-month-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold used to define peak months.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission_feature_rich_regularized_best.csv",
        help="Submission output CSV path.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="feature_rich_regularization",
        help="Optional run tag written to result metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = load_sales()
    folds = default_folds(profile="late_priority")

    configs = build_candidate_configs()
    if args.max_configs and args.max_configs > 0:
        configs = configs[: int(args.max_configs)]

    print(
        f"Loaded sales: {len(sales)} rows, "
        f"{sales.Date.min().date()} -> {sales.Date.max().date()}"
    )
    print(f"Evaluating {len(configs)} curated configs around refined5")

    result_rows: list[dict[str, Any]] = []
    fold_metric_frames: list[pd.DataFrame] = []
    oof_cache: dict[str, pd.DataFrame] = {}
    calibrator_cache: dict[str, Any] = {}

    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg.config_id} - {cfg.description}")
        oof = collect_feature_rich_oof(
            sales,
            folds=folds,
            params=cfg.params,
            random_state=42,
        )
        summary, fold_metrics, calibrator = evaluate_oof(
            oof,
            folds=folds,
            no_calibration=bool(args.no_calibration),
            peak_month_quantile=float(args.peak_month_quantile),
        )

        row = {
            "config_id": cfg.config_id,
            "description": cfg.description,
            "tag": args.tag,
            **summary,
            **{f"param_{k}": v for k, v in cfg.params.items()},
        }
        result_rows.append(row)

        fold_metrics = fold_metrics.copy()
        fold_metrics.insert(0, "config_id", cfg.config_id)
        fold_metric_frames.append(fold_metrics)

        oof_cache[cfg.config_id] = oof.copy()
        calibrator_cache[cfg.config_id] = calibrator

    results = pd.DataFrame(result_rows)
    best_payload, results_with_rules = select_best(
        results,
        no_calibration=bool(args.no_calibration),
        force_best_by_score=bool(args.force_best_by_score),
    )

    selected_id = str(best_payload["selected_config_id"])
    selected_cfg = next(c for c in configs if c.config_id == selected_id)
    selected_calibrator = calibrator_cache[selected_id]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    predict_submission(
        sales,
        params=selected_cfg.params,
        calibrator=selected_calibrator,
        no_calibration=bool(args.no_calibration),
        output=output_path,
        random_state=42,
    )

    selected_oof = oof_cache[selected_id].copy()
    selected_oof["prediction_calibrated"] = (
        selected_oof["prediction_raw"].to_numpy()
        if args.no_calibration or selected_calibrator is None
        else selected_calibrator.predict(selected_oof["prediction_raw"].to_numpy())
    )

    out_results = OUT_DIR / "feature_rich_regularization_results.csv"
    out_folds = OUT_DIR / "feature_rich_regularization_cv_by_fold.csv"
    out_best = OUT_DIR / "feature_rich_regularization_best_config.json"
    out_oof = OUT_DIR / "feature_rich_regularization_best_oof.csv"

    score_col = best_payload["score_column"]
    results_with_rules.sort_values([score_col, "calibrated_fold3_rmse"]).to_csv(
        out_results, index=False
    )
    pd.concat(fold_metric_frames, ignore_index=True).to_csv(out_folds, index=False)
    selected_oof.to_csv(out_oof, index=False)

    best_payload.update(
        {
            "tag": args.tag,
            "output_submission": str(output_path),
            "selected_params": selected_cfg.params,
            "refined5_params": REFINED5_PARAMS,
            "evaluated_config_count": len(configs),
        }
    )
    with out_best.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(best_payload), handle, indent=2, ensure_ascii=True)

    top_cols = [
        "config_id",
        score_col,
        "calibrated_fold3_rmse",
        "calibrated_peak_month_ape",
        "score_improve_pct",
        "rule_score_not_worse",
        "rule_fold3_not_worse",
        "rule_peak_not_worse",
        "rule_all",
    ]
    print("\nTop configs:")
    print(results_with_rules.sort_values(score_col)[top_cols].head(10).to_string(index=False))

    print("\nSelected:")
    print(json.dumps(_json_ready(best_payload), indent=2, ensure_ascii=True))

    print(f"\nWrote: {out_results}")
    print(f"Wrote: {out_folds}")
    print(f"Wrote: {out_best}")
    print(f"Wrote: {out_oof}")
    print(f"Wrote submission: {output_path}")


if __name__ == "__main__":
    main()
