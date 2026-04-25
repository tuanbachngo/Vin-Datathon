"""End-to-end driver: CV baselines + boosted ensembles, then write submissions."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.validation import default_folds, metrics
from src.baselines import (
    seasonal_naive_last_year,
    seasonal_naive_mean_2y,
    seasonal_naive_growth_adjusted,
)
from src.model import (
    predict_gbr,
    predict_hist_gbm,
    predict_lightgbm,
    predict_lightgbm_aux,
    predict_mlp,
    train_gbr,
    train_hist_gbm,
    train_lightgbm,
    train_lightgbm_aux,
    train_mlp,
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)


def load_sales() -> pd.DataFrame:
    s = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    return s.sort_values("Date").reset_index(drop=True)

def evaluate_cv(sales: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for fold in default_folds():
        train = sales[sales.Date <= fold.train_end]
        val = sales[fold.mask_val(sales.Date)]
        as_of = fold.train_end

        p_sn = seasonal_naive_last_year(val.Date, sales, as_of)
        p_mn = seasonal_naive_mean_2y(val.Date, sales, as_of)
        p_gr = seasonal_naive_growth_adjusted(val.Date, sales, as_of)

        hist_model, hist_feature_order = train_hist_gbm(train, as_of=as_of)
        gbr_model, gbr_feature_order = train_gbr(train, as_of=as_of)
        lightgbm_model, lightgbm_feature_order = train_lightgbm(train, as_of=as_of)
        lightgbm_aux_model, lightgbm_aux_feature_order = train_lightgbm_aux(
            train, as_of=as_of
        )
        lightgbm_all_aux_model, lightgbm_all_aux_feature_order = train_lightgbm_aux(
            train, as_of=as_of, selected_aux_features=None
        )
        mlp_model, mlp_feature_order = train_mlp(train, as_of=as_of)
        p_hist = predict_hist_gbm(
            hist_model, val.Date, sales, as_of, hist_feature_order
        )
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
        p_mlp = predict_mlp(
            mlp_model, val.Date, sales, as_of, mlp_feature_order
        )

        for name, pred in [
            ("seasonal_naive_lag365", p_sn),
            ("seasonal_naive_mean2y", p_mn),
            ("seasonal_naive_growth", p_gr),
            ("hist_gbm_log_target", p_hist),
            ("gbr_log_target", p_gbr),
            ("lightgbm_log_target", p_lgbm),
            ("lightgbm_top_aux_log_target", p_lgbm_aux),
            ("lightgbm_all_aux_log_target", p_lgbm_all_aux),
            ("mlp_deep_learning", p_mlp),
        ]:
            m = metrics(val.Revenue.values, pred)
            rows.append({"fold": fold.name, "model": name, **m})

    return pd.DataFrame(rows)


def fit_and_submit(
    sales: pd.DataFrame
) -> Path:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    as_of = sales.Date.max()

    hist_model, hist_feature_order = train_hist_gbm(sales, as_of=as_of)
    gbr_model, gbr_feature_order = train_gbr(sales, as_of=as_of)
    lightgbm_model, lightgbm_feature_order = train_lightgbm(sales, as_of=as_of)
    lightgbm_aux_model, lightgbm_aux_feature_order = train_lightgbm_aux(
        sales, as_of=as_of
    )
    lightgbm_all_aux_model, lightgbm_all_aux_feature_order = train_lightgbm_aux(
        sales, as_of=as_of, selected_aux_features=None
    )
    mlp_model, mlp_feature_order = train_mlp(sales, as_of=as_of)

    pred_hist = predict_hist_gbm(hist_model, sub.Date, sales, as_of, hist_feature_order)
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
    pred_mlp = predict_mlp(mlp_model, sub.Date, sales, as_of, mlp_feature_order)

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

    out_mlp = sub.copy()
    out_mlp["Revenue"] = pred_mlp
    out_mlp.to_csv(SUB_DIR / "submission_mlp_v4.csv", index=False)

    out_lgbm_aux.to_csv(SUB_DIR / "submission.csv", index=False)
    return SUB_DIR / "submission.csv"


def main() -> None:
    sales = load_sales()
    print(f"Loaded sales: {len(sales)} rows, "
          f"{sales.Date.min().date()} -> {sales.Date.max().date()}")

    print("\n=== CV on three walk-forward folds ===")
    cv = evaluate_cv(sales)
    print(cv.round(3).to_string(index=False))

    print("\n=== CV summary by model (mean across folds) ===")
    print(cv.groupby("model")[["MAE", "RMSE", "R2", "MAPE"]].mean().round(3))

    cv.to_csv(OUT_DIR / "cv_results.csv", index=False)

    print("\n=== Fit on all train, write submission ===")
    path = fit_and_submit(sales)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
