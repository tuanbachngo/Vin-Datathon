# EDA & Modelling Addendum — v4

This addendum captures the rerun after re-checking the assumptions in
`docs/EDA_modelling.md` and retuning the modelling stack.

## 1. Corrections to earlier assumptions

1. `lag_365` is not fully available across the 548-day horizon.
For the 2024 portion of the forecast it would require 2023 history, which is
not present in train. It is still useful for the 2023 slice, so the right fix
is to keep it as a partially observed feature and expose missingness to models
that cannot natively reason about `NaN`.

2. Heavy recency weighting is not helping.
The earlier pipeline downweighted old years. Re-running the same feature family
with flat weights improved CV, which means older history still contributes
useful seasonal shape once the model already has regime-aware seasonal lookup
features.

3. Same-calendar-date annual references are better than raw 365/730-day shifts.
Switching the baseline from raw day offsets to `DateOffset(years=1/2)` improved
the annual anchor and handled leap-year alignment more cleanly.

4. Auxiliary-table seasonal priors are not worth shipping.
Calendar-safe priors built from `orders`, `order_items`, `web_traffic`,
promotions, returns, and reviews looked plausible, but reduced CV. They are
mostly redundant with the direct revenue seasonal features and add noise.

## 2. Final preprocessing

- Target: `log(Revenue)`
- Features: calendar, Tet, annual seasonal lookup, long-lag revenue features
- Missingness: explicit `_missing` flags for partially observed lag features
- Imputation: median imputation only for the smooth `GradientBoostingRegressor`
- Scaling: none
- Sample weights: none in the final fit

## 3. Model choice

The winning stack is a static 3-way ensemble:

1. `HistGradientBoostingRegressor`
2. `GradientBoostingRegressor`
3. `seasonal_naive_mean2y`

Why this wins:

- `HistGBR` is strongest on the most recent holdout and handles `NaN`s natively.
- `GradientBoostingRegressor` is smoother and generalizes better on the exact
  18-month horizon fold.
- The seasonal baseline regularizes both tree models and cuts variance.

Additional experiments requested after v4:

- `LightGBMRegressor`
- a feed-forward neural network benchmark (`MLPRegressor` with scaling,
  imputation, and target quantile transform)
- auxiliary-table LightGBM branches using `orders`, `order_items`, `products`,
  `customers`, `geography`, `payments`, `shipments`, `returns`, `reviews`,
  `web_traffic`, `promotions`, and `inventory`

Neither replaced the current primary ensemble:

- LightGBM is competitive but still below the current GBR/HistGBR stack on mean CV.
- The neural network is usable as an experiment, but less stable and materially
  weaker than the tree models on this small tabular time-series problem.
- The auxiliary-table features are valid only after conversion to seasonal/lag
  priors because no auxiliary source covers 2023-2024. This analysis is captured
  in `docs/auxiliary_feature_analysis.md`.

## 4. Current CV results

Mean across the three walk-forward folds used by the pipeline:

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| `seasonal_naive_mean2y` | 705,507 | 963,072 | 0.634 | 0.271 |
| `hist_gbm_log_target` | 625,249 | 847,697 | 0.716 | 0.233 |
| `gbr_log_target` | 606,892 | 833,222 | 0.727 | 0.228 |
| `lightgbm_log_target` | 628,974 | 853,686 | 0.712 | 0.235 |
| `lightgbm_all_aux_log_target` | 636,496 | 867,219 | 0.704 | 0.239 |
| `lightgbm_top_aux_log_target` | 640,042 | 866,964 | 0.703 | 0.245 |
| `mlp_deep_learning` | 727,076 | 979,608 | 0.620 | 0.280 |
| `ensemble_hist_gbr_baseline` | **594,615** | **805,960** | **0.745** | **0.225** |

Per-fold ensemble R2:

| Fold | R2 |
|---|---:|
| `fold1_2020H2-2021` | 0.762 |
| `fold2_2021H2-2022` | 0.746 |
| `fold3_test_proxy` | 0.726 |

Final static ensemble weights tuned globally across all folds:

| Component | Weight |
|---|---:|
| `HistGradientBoostingRegressor` | 0.225 |
| `GradientBoostingRegressor` | 0.523 |
| `seasonal_naive_mean2y` | 0.252 |

## 5. Artifacts

| Path | Purpose |
|---|---|
| `outputs/cv_results.csv` | Fold-by-fold metrics for all v4 models |
| `outputs/ensemble_cv_results.csv` | Ensemble-only fold metrics |
| `outputs/ensemble_weights.csv` | Final global blend weights |
| `submissions/submission_hist_gbm_v4.csv` | HistGBR-only variant |
| `submissions/submission_gbr_v4.csv` | GBR-only variant |
| `submissions/submission_lightgbm_v4.csv` | LightGBM benchmark variant |
| `submissions/submission_lightgbm_all_aux_v5.csv` | LightGBM with all auxiliary priors |
| `submissions/submission_lightgbm_top_aux_v5.csv` | LightGBM with selected auxiliary priors |
| `submissions/submission_mlp_v4.csv` | Deep-learning benchmark variant |
| `submissions/submission_seasonal_mean2y_v4.csv` | Baseline variant |
| `submissions/submission_ensemble_v4.csv` | Primary tuned submission |
| `submissions/submission.csv` | Upload-ready copy of the primary submission |
