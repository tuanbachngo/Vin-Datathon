"""Forecast Revenue via decomposition: Revenue = Order Count x AOV.

Independent script. Reuses feature-rich no-lag residual XGBoost, trains two targets:
1) daily order_count
2) daily AOV = official sales Revenue / order_count
Then multiplies predictions and optionally applies one final total-level calibrator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .model import predict_xgboost_aux, train_xgboost_aux
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
    from model import predict_xgboost_aux, train_xgboost_aux
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _parse_float_list(text: str) -> list[float]:
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_daily_targets(*, target_floor: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = load_sales()
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])

    order_daily = (
        orders.groupby("order_date", as_index=False)
        .agg(order_count=("order_id", "nunique"))
        .rename(columns={"order_date": "Date"})
    )
    frame = sales[["Date", "Revenue"]].merge(order_daily, on="Date", how="left")
    frame["order_count"] = frame["order_count"].fillna(0.0).astype(float)
    frame["aov"] = frame["Revenue"] / np.maximum(frame["order_count"], 1.0)

    order_target = frame[["Date", "order_count"]].rename(columns={"order_count": "Revenue"})
    order_target["actual_order_count"] = frame["order_count"].to_numpy(dtype=float)
    order_target["Revenue"] = order_target["Revenue"].clip(lower=float(target_floor))

    aov_target = frame[["Date", "aov"]].rename(columns={"aov": "Revenue"})
    aov_target["actual_aov"] = frame["aov"].to_numpy(dtype=float)
    aov_target["Revenue"] = aov_target["Revenue"].clip(lower=float(target_floor))
    return sales, order_target, aov_target


def collect_component_oof(
    target_sales: pd.DataFrame,
    *,
    folds: list,
    params: dict[str, float | int | str],
    target_mode: str,
    target_floor: float,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    target_sales = target_sales.sort_values("Date").reset_index(drop=True)

    for fold in folds:
        train = target_sales[target_sales.Date <= fold.train_end].copy()
        val = target_sales[fold.mask_val(target_sales.Date)].sort_values("Date").reset_index(drop=True)
        as_of = fold.train_end

        model, feature_order = train_xgboost_aux(
            train[["Date", "Revenue"]],
            as_of=as_of,
            params=params,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode=target_mode,
            outlier_downweight=False,
            random_state=random_state,
        )
        pred = predict_xgboost_aux(
            model,
            val.Date,
            target_sales[["Date", "Revenue"]],
            as_of,
            feature_order,
            selected_aux_features=None,
            drop_lag_features=True,
            target_mode=target_mode,
        )
        pred = np.clip(np.asarray(pred, dtype=float), float(target_floor), None)
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "Date": val.Date.to_numpy(),
                    "actual_component": val["Revenue"].to_numpy(dtype=float),
                    "prediction_component": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _weighted_summary(fold_metrics: pd.DataFrame, folds: list) -> dict[str, float]:
    weight_map = {f.name: float(getattr(f, "weight", 1.0)) for f in folds}
    frame = fold_metrics.copy()
    frame["weight"] = frame["fold"].map(weight_map).fillna(1.0)
    w = frame["weight"].to_numpy(dtype=float)
    return {
        f"weighted_{col.lower()}": float(np.average(frame[col].to_numpy(dtype=float), weights=w))
        for col in ["MAE", "RMSE", "R2", "MAPE"]
    }


def evaluate_total_oof(
    official_sales: pd.DataFrame,
    order_oof: pd.DataFrame,
    aov_oof: pd.DataFrame,
    *,
    folds: list,
    peak_month_quantile: float,
    no_calibration: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any]:
    merged = order_oof.rename(
        columns={"actual_component": "actual_order_count", "prediction_component": "pred_order_count"}
    ).merge(
        aov_oof.rename(columns={"actual_component": "actual_aov", "prediction_component": "pred_aov"})[
            ["fold", "Date", "actual_aov", "pred_aov"]
        ],
        on=["fold", "Date"],
        how="inner",
    )
    merged = merged.merge(
        official_sales[["Date", "Revenue"]].rename(columns={"Revenue": "actual_revenue"}),
        on="Date",
        how="left",
    )
    merged["prediction_raw"] = (merged["pred_order_count"] * merged["pred_aov"]).clip(lower=0.0)

    calibrator = None
    merged["prediction_calibrated"] = merged["prediction_raw"].to_numpy(dtype=float)
    if not no_calibration:
        cal_result = fit_revenue_calibrator(
            merged[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]]
        )
        calibrator = cal_result.calibrator
        merged["prediction_calibrated"] = calibrator.predict(merged["prediction_raw"].to_numpy())

    fold_rows: list[dict[str, Any]] = []
    for fold_name, part in merged.groupby("fold", sort=False):
        raw_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_raw"].to_numpy())
        cal_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_calibrated"].to_numpy())
        fold_rows.append({"fold": fold_name, "prediction_type": "raw", **raw_m})
        fold_rows.append({"fold": fold_name, "prediction_type": "calibrated", **cal_m})
    fold_metrics = pd.DataFrame(fold_rows)
    raw_summary = _weighted_summary(fold_metrics[fold_metrics.prediction_type == "raw"], folds)
    cal_summary = _weighted_summary(fold_metrics[fold_metrics.prediction_type == "calibrated"], folds)

    fold3_raw = fold_metrics[(fold_metrics.fold == "fold3_test_proxy") & (fold_metrics.prediction_type == "raw")]
    fold3_cal = fold_metrics[(fold_metrics.fold == "fold3_test_proxy") & (fold_metrics.prediction_type == "calibrated")]

    _, peak_summary = build_peak_month_error_reports(
        merged[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]],
        calibrator=calibrator,
        quantile=float(peak_month_quantile),
    )
    peak_all = peak_summary[(peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "peak_month")]
    non_peak_all = peak_summary[(peak_summary["fold"] == "ALL") & (peak_summary["segment"] == "non_peak")]
    peak_row = peak_all.iloc[0] if not peak_all.empty else None
    non_peak_row = non_peak_all.iloc[0] if not non_peak_all.empty else None

    summary: dict[str, Any] = {
        **{f"raw_{k}": v for k, v in raw_summary.items()},
        **{f"calibrated_{k}": v for k, v in cal_summary.items()},
        "raw_fold3_rmse": float(fold3_raw.iloc[0]["RMSE"]) if not fold3_raw.empty else np.nan,
        "raw_fold3_mae": float(fold3_raw.iloc[0]["MAE"]) if not fold3_raw.empty else np.nan,
        "calibrated_fold3_rmse": float(fold3_cal.iloc[0]["RMSE"]) if not fold3_cal.empty else np.nan,
        "calibrated_fold3_mae": float(fold3_cal.iloc[0]["MAE"]) if not fold3_cal.empty else np.nan,
        "raw_peak_month_ape": float(peak_row["avg_monthly_ape_raw"]) if peak_row is not None else np.nan,
        "calibrated_peak_month_ape": float(peak_row["avg_monthly_ape_calibrated"]) if peak_row is not None else np.nan,
        "raw_non_peak_ape": float(non_peak_row["avg_monthly_ape_raw"]) if non_peak_row is not None else np.nan,
        "calibrated_non_peak_ape": float(non_peak_row["avg_monthly_ape_calibrated"]) if non_peak_row is not None else np.nan,
    }
    return merged, fold_metrics, summary, calibrator


def predict_component_full(
    target_sales: pd.DataFrame,
    dates: pd.Series,
    *,
    params: dict[str, float | int | str],
    target_mode: str,
    target_floor: float,
    random_state: int = 42,
) -> np.ndarray:
    target_sales = target_sales.sort_values("Date").reset_index(drop=True)
    as_of = pd.Timestamp(target_sales["Date"].max())
    model, feature_order = train_xgboost_aux(
        target_sales[["Date", "Revenue"]],
        as_of=as_of,
        params=params,
        selected_aux_features=None,
        drop_lag_features=True,
        target_mode=target_mode,
        outlier_downweight=False,
        random_state=random_state,
    )
    pred = predict_xgboost_aux(
        model,
        dates,
        target_sales[["Date", "Revenue"]],
        as_of,
        feature_order,
        selected_aux_features=None,
        drop_lag_features=True,
        target_mode=target_mode,
    )
    return np.clip(np.asarray(pred, dtype=float), float(target_floor), None)


def write_optional_blends(
    base_submission: pd.DataFrame,
    *,
    refined5_path: str,
    blend_weights: list[float],
    output_stem: str,
) -> list[Path]:
    if not refined5_path or not blend_weights:
        return []
    ref_path = Path(refined5_path)
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    if not ref_path.exists():
        raise FileNotFoundError(f"Refined5 submission not found: {ref_path}")

    refined = pd.read_csv(ref_path, parse_dates=["Date"])
    cand = base_submission.copy()
    cand["Date"] = pd.to_datetime(cand["Date"])
    if len(refined) != len(cand) or not (refined["Date"].reset_index(drop=True) == cand["Date"].reset_index(drop=True)).all():
        raise ValueError("Refined5 and candidate submission Date rows do not match.")

    out_paths: list[Path] = []
    for w in blend_weights:
        out = refined.copy()
        out["Revenue"] = (1.0 - w) * refined["Revenue"].astype(float) + w * cand["Revenue"].astype(float)
        out_path = ROOT / "submissions" / f"{output_stem}_blend_refined5_{int((1-w)*100):02d}_decomp_{int(w*100):02d}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast Revenue as order_count x AOV.")
    parser.add_argument("--output", type=str, default="submissions/submission_orders_x_aov.csv")
    parser.add_argument("--cv-profile", choices=["late_priority", "legacy"], default="late_priority")
    parser.add_argument("--target-mode", choices=["residual", "direct"], default="residual")
    parser.add_argument("--target-floor", type=float, default=1.0)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--peak-month-quantile", type=float, default=0.75)
    parser.add_argument("--refined5-submission", type=str, default="")
    parser.add_argument("--blend-weights", type=str, default="0.1,0.2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    folds = default_folds(profile=args.cv_profile)
    official_sales, order_target, aov_target = build_daily_targets(target_floor=float(args.target_floor))

    print(f"Loaded official sales: {len(official_sales)} rows")
    print(f"Target mode: {args.target_mode}; calibration: {not args.no_calibration}")
    print("Collecting OOF for order_count component...")
    order_oof = collect_component_oof(
        order_target,
        folds=folds,
        params=REFINED5_PARAMS,
        target_mode=str(args.target_mode),
        target_floor=float(args.target_floor),
    )
    print("Collecting OOF for AOV component...")
    aov_oof = collect_component_oof(
        aov_target,
        folds=folds,
        params=REFINED5_PARAMS,
        target_mode=str(args.target_mode),
        target_floor=float(args.target_floor),
    )

    total_oof, fold_metrics, summary, calibrator = evaluate_total_oof(
        official_sales,
        order_oof,
        aov_oof,
        folds=folds,
        peak_month_quantile=float(args.peak_month_quantile),
        no_calibration=bool(args.no_calibration),
    )
    print("\nOOF summary:")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=True))

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    print("\nTraining full order_count and AOV models, writing submission...")
    pred_orders = predict_component_full(
        order_target,
        sub.Date,
        params=REFINED5_PARAMS,
        target_mode=str(args.target_mode),
        target_floor=float(args.target_floor),
    )
    pred_aov = predict_component_full(
        aov_target,
        sub.Date,
        params=REFINED5_PARAMS,
        target_mode=str(args.target_mode),
        target_floor=float(args.target_floor),
    )
    pred_raw = np.clip(pred_orders * pred_aov, 0.0, None)
    pred = pred_raw if args.no_calibration or calibrator is None else calibrator.predict(pred_raw)

    out = sub.copy()
    out["Revenue"] = pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    OUT_DIR.mkdir(exist_ok=True)
    total_oof.to_csv(OUT_DIR / "orders_aov_oof.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "orders_aov_cv_by_fold.csv", index=False)
    with (OUT_DIR / "orders_aov_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(
                {
                    "output_submission": str(output_path),
                    "params": REFINED5_PARAMS,
                    "target_mode": args.target_mode,
                    "target_floor": args.target_floor,
                    "no_calibration": bool(args.no_calibration),
                    **summary,
                }
            ),
            handle,
            indent=2,
            ensure_ascii=True,
        )

    blend_paths = write_optional_blends(
        out,
        refined5_path=args.refined5_submission,
        blend_weights=_parse_float_list(args.blend_weights),
        output_stem=output_path.stem,
    )

    print(f"\nWrote submission: {output_path}")
    print("Wrote diagnostics:")
    print(f"  {OUT_DIR / 'orders_aov_oof.csv'}")
    print(f"  {OUT_DIR / 'orders_aov_cv_by_fold.csv'}")
    print(f"  {OUT_DIR / 'orders_aov_summary.json'}")
    if blend_paths:
        print("Wrote optional refined5 blends:")
        for p in blend_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
