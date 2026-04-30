"""Regime-weighted refined5 XGBoost experiment (no-lag residual, feature-rich).

Goal:
- Keep refined5 feature-rich setup (selected_aux_features=None).
- Train with regime-based sample weights.
- Evaluate on late-priority walk-forward folds with isotonic calibration.
- Select best profile using CV guard vs baseline profile.
- Train full model and export submission + scaled variants.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # package import style
    from .calibration import RevenueCalibrator, fit_revenue_calibrator
    from .model import (
        XGBRegressor,
        _baseline_prediction_array,
        _build_xgboost_aux_matrix,
        predict_xgboost_aux,
    )
    from .run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
        build_peak_month_error_reports,
        default_folds,
        load_sales,
        metrics,
    )
except ImportError:  # script style: python src/train_regime_weighted_refined5.py
    from calibration import RevenueCalibrator, fit_revenue_calibrator
    from model import (
        XGBRegressor,
        _baseline_prediction_array,
        _build_xgboost_aux_matrix,
        predict_xgboost_aux,
    )
    from run_pipeline import (
        DATA_DIR,
        OUT_DIR,
        ROOT,
        SUB_DIR,
        build_peak_month_error_reports,
        default_folds,
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
class RegimeProfile:
    profile_id: str
    weight_2012_2016: float
    weight_2017_2019: float
    weight_2020_2022: float


def build_profiles() -> list[RegimeProfile]:
    return [
        RegimeProfile("baseline", 1.0, 1.0, 1.0),
        RegimeProfile("balanced_recovery", 0.7, 1.4, 1.0),
        RegimeProfile("strong_recovery", 0.6, 1.7, 0.9),
        RegimeProfile("aggressive_recovery", 0.5, 2.0, 0.5),
        RegimeProfile("conservative_recovery", 0.8, 1.25, 1.0),
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def regime_sample_weights(
    dates: pd.Series | pd.DatetimeIndex,
    profile: RegimeProfile,
) -> np.ndarray:
    years = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
    w = np.full(len(years), float(profile.weight_2020_2022), dtype=float)
    w = np.where(years <= 2016, float(profile.weight_2012_2016), w)
    w = np.where((years >= 2017) & (years <= 2019), float(profile.weight_2017_2019), w)
    w = np.where((years >= 2020) & (years <= 2022), float(profile.weight_2020_2022), w)
    return w


def train_xgboost_aux_regime_weighted(
    sales_train: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    params: dict[str, float | int | str],
    profile: RegimeProfile,
    random_state: int = 42,
) -> tuple[object, list[str]]:
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Run `.venv\\Scripts\\python.exe -m pip install xgboost`."
        )

    X = _build_xgboost_aux_matrix(
        sales_train.Date,
        sales_train,
        as_of,
        selected_aux_features=None,  # refined5 feature-rich
        drop_lag_features=True,      # no-lag
        target_mode="residual",
        baseline_fn=None,
    )
    y = np.log(sales_train.Revenue.to_numpy(dtype=float))
    baseline = _baseline_prediction_array(None, sales_train.Date, sales_train, as_of)
    valid = np.isfinite(baseline)
    if not valid.any():
        raise RuntimeError("Residual baseline produced no valid rows for training.")

    X_fit = X.loc[valid].reset_index(drop=True)
    y_fit = y[valid] - np.log(np.clip(baseline[valid], 1e-6, None))
    sw = regime_sample_weights(sales_train.Date, profile)[valid]

    model = XGBRegressor(**params, random_state=random_state)
    model.fit(X_fit, y_fit, sample_weight=sw)
    return model, list(X_fit.columns)


def collect_oof_for_profile(
    sales: pd.DataFrame,
    *,
    folds: list,
    params: dict[str, float | int | str],
    profile: RegimeProfile,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        train = sales[sales.Date <= fold.train_end].copy()
        val = sales[fold.mask_val(sales.Date)].sort_values("Date").reset_index(drop=True)
        as_of = fold.train_end

        model, feature_order = train_xgboost_aux_regime_weighted(
            train,
            as_of=as_of,
            params=params,
            profile=profile,
            random_state=random_state,
        )
        pred = predict_xgboost_aux(
            model,
            val.Date,
            sales,
            as_of,
            feature_order,
            selected_aux_features=None,  # refined5 feature-rich
            drop_lag_features=True,      # no-lag
            target_mode="residual",
        )
        rows.append(
            pd.DataFrame(
                {
                    "profile_id": profile.profile_id,
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "Date": val.Date.to_numpy(),
                    "actual_revenue": val.Revenue.to_numpy(),
                    "prediction_raw": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _weighted_from_folds(
    fold_rows: list[dict[str, Any]],
    folds: list,
    *,
    prefix: str,
) -> dict[str, float]:
    fold_weight = {f.name: float(getattr(f, "weight", 1.0)) for f in folds}
    weights = np.asarray([fold_weight.get(str(r["fold"]), 1.0) for r in fold_rows], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(fold_rows), dtype=float)

    out: dict[str, float] = {}
    for metric_name in ["MAE", "RMSE", "R2", "MAPE"]:
        vals = np.asarray([float(r[metric_name]) for r in fold_rows], dtype=float)
        out[f"{prefix}_weighted_{metric_name.lower()}"] = float(
            np.average(vals, weights=weights)
        )
    return out


def evaluate_profile_oof(
    oof: pd.DataFrame,
    *,
    folds: list,
    peak_month_quantile: float,
) -> tuple[dict[str, Any], pd.DataFrame, RevenueCalibrator]:
    cal_result = fit_revenue_calibrator(
        oof[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]]
    )
    calibrator = cal_result.calibrator

    frame = oof.copy()
    frame["prediction_calibrated"] = calibrator.predict(frame["prediction_raw"].to_numpy())

    raw_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for fold_name, part in frame.groupby("fold", sort=False):
        raw_m = metrics(part["actual_revenue"].to_numpy(), part["prediction_raw"].to_numpy())
        cal_m = metrics(
            part["actual_revenue"].to_numpy(),
            part["prediction_calibrated"].to_numpy(),
        )
        raw_rows.append({"fold": fold_name, **raw_m})
        cal_rows.append({"fold": fold_name, **cal_m})
        fold_metric_rows.append({"fold": fold_name, "prediction_type": "raw", **raw_m})
        fold_metric_rows.append(
            {"fold": fold_name, "prediction_type": "calibrated", **cal_m}
        )

    raw_weighted = _weighted_from_folds(raw_rows, folds, prefix="raw")
    cal_weighted = _weighted_from_folds(cal_rows, folds, prefix="calibrated")

    fold3_raw = next((r for r in raw_rows if r["fold"] == "fold3_test_proxy"), None)
    fold3_cal = next((r for r in cal_rows if r["fold"] == "fold3_test_proxy"), None)
    if fold3_raw is None or fold3_cal is None:
        raise RuntimeError("fold3_test_proxy not found in evaluation.")

    _, peak_summary = build_peak_month_error_reports(
        oof[["fold", "train_end", "Date", "actual_revenue", "prediction_raw"]],
        calibrator=calibrator,
        quantile=float(peak_month_quantile),
    )
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
            float(non_peak_row["avg_monthly_ape_calibrated"])
            if non_peak_row is not None
            else np.nan
        ),
    }
    return summary, pd.DataFrame(fold_metric_rows), calibrator


def select_best_profile(results: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    df = results.copy()
    base = df[df["profile_id"] == "baseline"]
    if base.empty:
        raise RuntimeError("Baseline profile row missing.")
    b = base.iloc[0]

    baseline_rmse = float(b["calibrated_weighted_rmse"])
    baseline_fold3 = float(b["calibrated_fold3_rmse"])
    baseline_peak = float(b["calibrated_peak_month_ape"])

    df["rule_rmse_better_than_baseline"] = df["calibrated_weighted_rmse"] < baseline_rmse
    df["rule_fold3_not_worse"] = df["calibrated_fold3_rmse"] <= baseline_fold3
    df["rule_peak_not_worse"] = df["calibrated_peak_month_ape"] <= baseline_peak
    df["rule_all"] = (
        (df["profile_id"] != "baseline")
        & df["rule_rmse_better_than_baseline"]
        & df["rule_fold3_not_worse"]
        & df["rule_peak_not_worse"]
    )
    df["rmse_improve_pct_vs_baseline"] = (
        (baseline_rmse - df["calibrated_weighted_rmse"]) / max(baseline_rmse, 1e-12)
    )

    ranked = df.sort_values(
        ["calibrated_weighted_rmse", "calibrated_fold3_rmse"], ascending=[True, True]
    ).reset_index(drop=True)
    eligible = ranked[ranked["rule_all"]]

    if eligible.empty:
        selected_id = "baseline"
        status = "baseline_kept_no_guard_winner"
        reason = "no_profile_passed_rmse_plus_fold3_plus_peak_guards"
    else:
        selected_id = str(eligible.iloc[0]["profile_id"])
        status = "guard_winner_selected"
        reason = "profile_passed_guards_and_improved_weighted_rmse"

    selected_row = df[df["profile_id"] == selected_id].iloc[0].to_dict()
    payload = {
        "status": status,
        "selection_reason": reason,
        "baseline_profile_id": "baseline",
        "baseline_calibrated_weighted_rmse": baseline_rmse,
        "baseline_calibrated_fold3_rmse": baseline_fold3,
        "baseline_calibrated_peak_month_ape": baseline_peak,
        "selected_profile_id": selected_id,
        "selected_profile": selected_row,
    }
    return payload, df


def predict_full_submission(
    sales: pd.DataFrame,
    *,
    params: dict[str, float | int | str],
    profile: RegimeProfile,
    calibrator: RevenueCalibrator,
    output_path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = pd.Timestamp(sales["Date"].max())
    model, feature_order = train_xgboost_aux_regime_weighted(
        sales,
        as_of=as_of,
        params=params,
        profile=profile,
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
    pred_cal = calibrator.predict(pred_raw)

    out = sub.copy()
    out["Revenue"] = pred_cal
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def write_scaled_variants(base_submission: pd.DataFrame, base_path: Path) -> list[str]:
    written: list[str] = []
    for scale in [1.03, 1.04, 1.05]:
        suffix = str(scale).replace(".", "")
        out_path = base_path.with_name(
            f"{base_path.stem}_x{suffix}{base_path.suffix}"
        )
        df = base_submission.copy()
        df["Revenue"] = df["Revenue"].astype(float) * float(scale)
        df.to_csv(out_path, index=False)
        written.append(str(out_path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-weighted refined5 feature-rich XGBoost evaluation."
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=0,
        help="Optional debug limit. 0 = evaluate all profiles.",
    )
    parser.add_argument(
        "--peak-month-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for peak-month segmentation.",
    )
    parser.add_argument(
        "--submission-output",
        type=str,
        default="submissions/submission_regime_weighted_refined5.csv",
        help="Legacy selected-profile output path (kept for compatibility).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = load_sales()
    folds = default_folds(profile="late_priority")
    profiles = build_profiles()
    if int(args.max_profiles) > 0:
        profiles = profiles[: int(args.max_profiles)]

    print(
        f"Loaded sales: {len(sales)} rows, "
        f"{sales.Date.min().date()} -> {sales.Date.max().date()}"
    )
    print(f"Evaluating {len(profiles)} regime profiles")

    result_rows: list[dict[str, Any]] = []
    fold_metric_frames: list[pd.DataFrame] = []
    oof_cache: dict[str, pd.DataFrame] = {}
    calibrator_cache: dict[str, RevenueCalibrator] = {}

    for idx, profile in enumerate(profiles, start=1):
        print(f"[{idx}/{len(profiles)}] profile={profile.profile_id}")
        oof = collect_oof_for_profile(
            sales,
            folds=folds,
            params=REFINED5_PARAMS,
            profile=profile,
            random_state=42,
        )
        summary, fold_metrics, calibrator = evaluate_profile_oof(
            oof,
            folds=folds,
            peak_month_quantile=float(args.peak_month_quantile),
        )

        row = {
            "profile_id": profile.profile_id,
            "w_2012_2016": profile.weight_2012_2016,
            "w_2017_2019": profile.weight_2017_2019,
            "w_2020_2022": profile.weight_2020_2022,
            **summary,
        }
        result_rows.append(row)

        fold_metrics = fold_metrics.copy()
        fold_metrics.insert(0, "profile_id", profile.profile_id)
        fold_metric_frames.append(fold_metrics)

        oof_cache[profile.profile_id] = oof.copy()
        calibrator_cache[profile.profile_id] = calibrator

    results = pd.DataFrame(result_rows)
    best_payload, results_with_rules = select_best_profile(results)
    selected_profile_id = str(best_payload["selected_profile_id"])
    selected_profile = next(p for p in profiles if p.profile_id == selected_profile_id)
    selected_calibrator = calibrator_cache[selected_profile_id]

    oof_all_profiles: list[pd.DataFrame] = []
    for p in profiles:
        p_oof = oof_cache[p.profile_id].copy()
        p_cal = calibrator_cache[p.profile_id]
        p_oof["prediction_calibrated"] = p_cal.predict(p_oof["prediction_raw"].to_numpy())
        oof_all_profiles.append(p_oof)

    export_manifest: list[dict[str, Any]] = []
    for p in profiles:
        profile_calibrator = calibrator_cache[p.profile_id]
        profile_sub_path = SUB_DIR / f"submission_regime_{p.profile_id}.csv"
        profile_submission = predict_full_submission(
            sales,
            params=REFINED5_PARAMS,
            profile=p,
            calibrator=profile_calibrator,
            output_path=profile_sub_path,
            random_state=42,
        )
        scaled_paths = write_scaled_variants(profile_submission, profile_sub_path)
        export_manifest.append(
            {
                "profile_id": p.profile_id,
                "submission_output": str(profile_sub_path),
                "scaled_submission_outputs": scaled_paths,
            }
        )

    # Legacy selected-profile output retained for backward compatibility.
    sub_path = Path(args.submission_output)
    if not sub_path.is_absolute():
        sub_path = ROOT / sub_path
    selected_submission = predict_full_submission(
        sales,
        params=REFINED5_PARAMS,
        profile=selected_profile,
        calibrator=selected_calibrator,
        output_path=sub_path,
        random_state=42,
    )
    selected_scaled_paths = write_scaled_variants(selected_submission, sub_path)

    out_results = OUT_DIR / "regime_weighted_results.csv"
    out_best = OUT_DIR / "regime_weighted_best_config.json"
    out_oof = OUT_DIR / "regime_weighted_oof.csv"

    results_with_rules.sort_values(
        ["calibrated_weighted_rmse", "calibrated_fold3_rmse"], ascending=[True, True]
    ).to_csv(out_results, index=False)
    pd.concat(oof_all_profiles, ignore_index=True).to_csv(out_oof, index=False)

    best_payload.update(
        {
            "refined5_params": REFINED5_PARAMS,
            "evaluated_profile_count": len(profiles),
            "submission_output": str(sub_path),
            "scaled_submission_outputs": selected_scaled_paths,
            "exported_profile_submissions": export_manifest,
            "fold_profile": "late_priority",
        }
    )
    with out_best.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(best_payload), handle, indent=2, ensure_ascii=True)

    print("\nTop profiles by calibrated weighted RMSE:")
    show_cols = [
        "profile_id",
        "calibrated_weighted_rmse",
        "calibrated_weighted_mae",
        "calibrated_fold3_rmse",
        "calibrated_fold3_mae",
        "calibrated_peak_month_ape",
        "calibrated_non_peak_ape",
        "rule_rmse_better_than_baseline",
        "rule_fold3_not_worse",
        "rule_peak_not_worse",
        "rule_all",
    ]
    print(
        results_with_rules.sort_values("calibrated_weighted_rmse")[show_cols]
        .head(10)
        .to_string(index=False)
    )

    print("\nSelected profile summary:")
    print(json.dumps(_json_ready(best_payload), indent=2, ensure_ascii=True))

    print(f"\nWrote: {out_results}")
    print(f"Wrote: {out_best}")
    print(f"Wrote: {out_oof}")
    print(f"Wrote legacy selected submission: {sub_path}")
    for p in selected_scaled_paths:
        print(f"Wrote legacy selected scaled submission: {p}")
    for item in export_manifest:
        print(f"Wrote profile submission: {item['submission_output']}")
        for p in item["scaled_submission_outputs"]:
            print(f"Wrote profile scaled submission: {p}")


if __name__ == "__main__":
    main()
