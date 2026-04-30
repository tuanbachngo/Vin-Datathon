"""End-to-end driver: CV baselines + boosted ensembles, then write submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from baselines import (
    seasonal_naive_growth_adjusted,
    seasonal_naive_last_year,
    seasonal_naive_mean_2y,
)
from calibration import RevenueCalibrator, fit_revenue_calibrator
from features import (
    LOOKUP_HISTORY_MODE_FULL_HISTORY,
    LOOKUP_HISTORY_MODE_LEGACY_STRUCTURE_RECENT_LEVEL,
    LOOKUP_HISTORY_MODE_RECENT_REGIME,
    LOOKUP_HISTORY_MODES,
)
from model import (
    TOP_AUX_FEATURES,
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
from xgb_feature_selection import select_aux_features_via_mic

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
        "mic_scores": OUT_DIR / f"{prefix}_mic_scores.csv",
        "mic_group_summary": OUT_DIR / f"{prefix}_mic_group_summary.csv",
        "mic_summary": OUT_DIR / f"{prefix}_mic_summary.json",
        "mic_selected": OUT_DIR / f"{prefix}_mic_selected_features.json",
    }


def xgb_runtime_config(
    no_lag: bool,
    no_lag_residual: bool,
    *,
    lookup_history_mode: str = LOOKUP_HISTORY_MODE_RECENT_REGIME,
    use_mic_features: bool = False,
) -> dict[str, str | bool]:
    if no_lag_residual:
        config: dict[str, str | bool] = {
            "artifact_prefix": "xgboost_no_lag_residual",
            "model_name": "xgboost_top_aux_no_lag_residual",
            "submission_prefix": "submission_xgboost_top_aux_no_lag_residual",
            "drop_lag_features": True,
            "target_mode": "residual",
        }
    elif no_lag:
        config = {
            "artifact_prefix": "xgboost_no_lag",
            "model_name": "xgboost_top_aux_no_lag",
            "submission_prefix": "submission_xgboost_top_aux_no_lag",
            "drop_lag_features": True,
            "target_mode": "direct",
        }
    else:
        config = {
            "artifact_prefix": "xgboost",
            "model_name": "xgboost_top_aux_log_target",
            "submission_prefix": "submission_xgboost_top_aux",
            "drop_lag_features": False,
            "target_mode": "direct",
        }

    suffixes: list[str] = []
    if lookup_history_mode == LOOKUP_HISTORY_MODE_LEGACY_STRUCTURE_RECENT_LEVEL:
        suffixes.append("legacy_lookup")
    elif lookup_history_mode == LOOKUP_HISTORY_MODE_FULL_HISTORY:
        suffixes.append("full_history_lookup")
    if use_mic_features:
        suffixes.append("mic")
    if suffixes:
        suffix = "_".join(suffixes)
        for key in ("artifact_prefix", "model_name", "submission_prefix"):
            config[key] = f"{config[key]}_{suffix}"
    return config


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
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_xgboost_params(
    params: dict[str, float | int | str],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready_dict(params), handle, indent=2, ensure_ascii=True)


def save_calibration_summary(
    summary: dict[str, float],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready_dict(summary), handle, indent=2, ensure_ascii=True)


def save_json_payload(
    payload: object,
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def evaluate_cv(
    sales: pd.DataFrame,
    *,
    xgb_params: dict[str, float | int | str] | None = None,
    xgb_oof: pd.DataFrame | None = None,
    xgb_calibrator: RevenueCalibrator | None = None,
    xgb_selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    xgb_drop_lag_features: bool = False,
    xgb_target_mode: str = "direct",
    xgb_lookup_history_mode: str = LOOKUP_HISTORY_MODE_RECENT_REGIME,
    xgb_model_name: str = "xgboost_top_aux_log_target",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    xgb_by_fold = None
    if xgb_oof is not None:
        xgb_by_fold = {
            fold_name: frame.sort_values("Date").reset_index(drop=True)
            for fold_name, frame in xgb_oof.groupby("fold", sort=False)
        }

    for fold in default_folds():
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)]
        as_of = fold.train_end

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

        if xgb_by_fold is None:
            xgb_model, xgb_feature_order = train_xgboost_aux(
                train,
                as_of=as_of,
                params=xgb_params,
                selected_aux_features=xgb_selected_aux_features,
                drop_lag_features=xgb_drop_lag_features,
                target_mode=xgb_target_mode,
                lookup_history_mode=xgb_lookup_history_mode,
            )
            p_xgb = predict_xgboost_aux(
                xgb_model,
                val.Date,
                sales,
                as_of,
                xgb_feature_order,
                selected_aux_features=xgb_selected_aux_features,
                drop_lag_features=xgb_drop_lag_features,
                target_mode=xgb_target_mode,
                lookup_history_mode=xgb_lookup_history_mode,
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

        model_predictions = [
            ("seasonal_naive_lag365", p_sn),
            ("seasonal_naive_mean2y", p_mn),
            ("seasonal_naive_growth", p_gr),
            ("gbr_log_target", p_gbr),
            ("lightgbm_log_target", p_lgbm),
            ("lightgbm_top_aux_log_target", p_lgbm_aux),
            ("lightgbm_all_aux_log_target", p_lgbm_all_aux),
            (xgb_model_name, p_xgb),
            ("mlp_deep_learning", p_mlp),
        ]
        if p_hist is not None:
            model_predictions.insert(3, ("hist_gbm_log_target", p_hist))

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
    xgb_selected_aux_features: list[str] | None = TOP_AUX_FEATURES,
    xgb_drop_lag_features: bool = False,
    xgb_target_mode: str = "direct",
    xgb_lookup_history_mode: str = LOOKUP_HISTORY_MODE_RECENT_REGIME,
    xgb_submission_prefix: str = "submission_xgboost_top_aux",
    write_submission_alias: bool = True,
) -> Path:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = sales.Date.max()

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
    xgb_model, xgb_feature_order = train_xgboost_aux(
        sales,
        as_of=as_of,
        params=xgb_params,
        selected_aux_features=xgb_selected_aux_features,
        drop_lag_features=xgb_drop_lag_features,
        target_mode=xgb_target_mode,
        lookup_history_mode=xgb_lookup_history_mode,
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
    pred_xgb = predict_xgboost_aux(
        xgb_model,
        sub.Date,
        sales,
        as_of,
        xgb_feature_order,
        selected_aux_features=xgb_selected_aux_features,
        drop_lag_features=xgb_drop_lag_features,
        target_mode=xgb_target_mode,
        lookup_history_mode=xgb_lookup_history_mode,
    )
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
    out_xgb.to_csv(SUB_DIR / f"{xgb_submission_prefix}_raw.csv", index=False)

    out_xgb_calibrated = sub.copy()
    out_xgb_calibrated["Revenue"] = pred_xgb_calibrated
    out_xgb_calibrated.to_csv(
        SUB_DIR / f"{xgb_submission_prefix}_calibrated.csv", index=False
    )

    out_mlp = sub.copy()
    out_mlp["Revenue"] = pred_mlp
    out_mlp.to_csv(SUB_DIR / "submission_mlp_v4.csv", index=False)

    if write_submission_alias:
        out_xgb_calibrated.to_csv(SUB_DIR / "submission.csv", index=False)
        return SUB_DIR / "submission.csv"
    return SUB_DIR / f"{xgb_submission_prefix}_calibrated.csv"


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
        "--xgb-lookup-history-mode",
        choices=LOOKUP_HISTORY_MODES,
        default=LOOKUP_HISTORY_MODE_RECENT_REGIME,
        help=(
            "Seasonal lookup history mode for XGBoost feature/baseline construction. "
            "`legacy_structure_recent_level` keeps pre-2020 seasonal structure while "
            "still scaling levels with the latest 12 months."
        ),
    )
    parser.add_argument(
        "--xgb-mic-select",
        action="store_true",
        help=(
            "Build a MIC-ranked auxiliary feature list from train folds before "
            "tuning/CV/submission and use it as selected_aux_features."
        ),
    )
    parser.add_argument(
        "--xgb-mic-top-n",
        type=int,
        default=len(TOP_AUX_FEATURES),
        help="Number of MIC-ranked auxiliary features to keep when --xgb-mic-select is enabled.",
    )
    parser.add_argument(
        "--xgb-mic-family-top-k",
        type=int,
        default=2,
        help="Maximum MIC-selected variants to keep per raw auxiliary series.",
    )
    parser.add_argument(
        "--xgb-mic-stability-threshold",
        type=float,
        default=0.15,
        help=(
            "Minimum pre-2020 vs post-2020 MIC ratio for stability filtering. "
            "Features with missing regime coverage bypass this filter."
        ),
    )
    parser.add_argument(
        "--xgb-mic-min-regime-rows",
        type=int,
        default=180,
        help="Minimum rows required inside a regime subset before MIC stability is computed.",
    )
    parser.add_argument(
        "--xgb-mic-max-bins",
        type=int,
        default=8,
        help="Maximum quantile bin count for the MIC approximation backend.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sales = load_sales()
    print(
        f"Loaded sales: {len(sales)} rows, "
        f"{sales.Date.min().date()} -> {sales.Date.max().date()}"
    )

    xgb_config = xgb_runtime_config(
        args.xgb_no_lag,
        args.xgb_no_lag_residual,
        lookup_history_mode=str(args.xgb_lookup_history_mode),
        use_mic_features=bool(args.xgb_mic_select),
    )
    xgb_paths = xgb_artifact_paths(str(xgb_config["artifact_prefix"]))
    xgb_params = load_xgboost_params(xgb_paths["params"])
    xgb_source = "defaults"
    xgb_selected_aux_features: list[str] | None = TOP_AUX_FEATURES

    if args.xgb_mic_select:
        print(
            "\n=== MIC auxiliary feature selection "
            f"({xgb_config['artifact_prefix']}, top_n={args.xgb_mic_top_n}) ==="
        )
        mic_result = select_aux_features_via_mic(
            sales,
            drop_lag_features=bool(xgb_config["drop_lag_features"]),
            target_mode=str(xgb_config["target_mode"]),
            lookup_history_mode=str(args.xgb_lookup_history_mode),
            top_n=int(args.xgb_mic_top_n),
            family_top_k=int(args.xgb_mic_family_top_k),
            stability_threshold=float(args.xgb_mic_stability_threshold),
            min_regime_rows=int(args.xgb_mic_min_regime_rows),
            max_bins=int(args.xgb_mic_max_bins),
        )
        xgb_selected_aux_features = mic_result.selected_features
        mic_result.feature_scores.to_csv(xgb_paths["mic_scores"], index=False)
        mic_result.group_summary.to_csv(xgb_paths["mic_group_summary"], index=False)
        save_json_payload(mic_result.summary, xgb_paths["mic_summary"])
        save_json_payload(mic_result.selected_features, xgb_paths["mic_selected"])
        print(
            f"Selected {len(xgb_selected_aux_features)} aux features via "
            f"{mic_result.summary['backend']}:"
        )
        print(", ".join(xgb_selected_aux_features))

    if args.tune_xgboost:
        print(
            "\n=== Optuna tuning XGBoost "
            f"({xgb_config['artifact_prefix']}, {args.xgb_trials} trials) ==="
        )
        tuning_result = tune_xgboost_hyperparameters(
            sales,
            n_trials=args.xgb_trials,
            selected_aux_features=xgb_selected_aux_features,
            drop_lag_features=bool(xgb_config["drop_lag_features"]),
            target_mode=str(xgb_config["target_mode"]),
            lookup_history_mode=str(args.xgb_lookup_history_mode),
        )
        xgb_params = tuning_result.best_params
        xgb_source = f"optuna_{args.xgb_trials}_trials"
        save_xgboost_params(xgb_params, xgb_paths["params"])
        tuning_result.trials.to_csv(xgb_paths["trials"], index=False)
        print(f"Best RMSE: {tuning_result.best_value:.3f}")
        print(f"Saved params to {xgb_paths['params']}")
    elif xgb_params is not None:
        xgb_source = xgb_paths["params"].name

    print(f"\n=== XGBoost parameter source: {xgb_source} ===")
    xgb_oof = collect_xgboost_oof_predictions(
        sales,
        params=xgb_params,
        selected_aux_features=xgb_selected_aux_features,
        drop_lag_features=bool(xgb_config["drop_lag_features"]),
        target_mode=str(xgb_config["target_mode"]),
        lookup_history_mode=str(args.xgb_lookup_history_mode),
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

    print("\n=== CV on three walk-forward folds ===")
    cv = evaluate_cv(
        sales,
        xgb_params=xgb_params,
        xgb_oof=xgb_oof,
        xgb_calibrator=calibration_result.calibrator,
        xgb_selected_aux_features=xgb_selected_aux_features,
        xgb_drop_lag_features=bool(xgb_config["drop_lag_features"]),
        xgb_target_mode=str(xgb_config["target_mode"]),
        xgb_lookup_history_mode=str(args.xgb_lookup_history_mode),
        xgb_model_name=str(xgb_config["model_name"]),
    )
    print(cv.round(3).to_string(index=False))

    print("\n=== CV summary by model (mean across folds) ===")
    print(cv.groupby("model")[["MAE", "RMSE", "R2", "MAPE"]].mean().round(3))
    cv.to_csv(xgb_paths["cv"], index=False)

    print("\n=== Fit on all train, write submission ===")
    path = fit_and_submit(
        sales,
        xgb_params=xgb_params,
        xgb_calibrator=calibration_result.calibrator,
        xgb_selected_aux_features=xgb_selected_aux_features,
        xgb_drop_lag_features=bool(xgb_config["drop_lag_features"]),
        xgb_target_mode=str(xgb_config["target_mode"]),
        xgb_lookup_history_mode=str(args.xgb_lookup_history_mode),
        xgb_submission_prefix=str(xgb_config["submission_prefix"]),
        write_submission_alias=not (args.xgb_no_lag_residual or args.xgb_no_lag),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
