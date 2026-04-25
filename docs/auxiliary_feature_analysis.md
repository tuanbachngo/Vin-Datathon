# Auxiliary Table Feature Analysis

This note documents the rerun that explicitly used the non-`sales.csv` tables:
`orders`, `order_items`, `products`, `customers`, `geography`, `payments`,
`shipments`, `returns`, `reviews`, `web_traffic`, `promotions`, and `inventory`.

## 1. Forecast constraint

All auxiliary tables end at or before `2022-12-31`, while the target horizon is
`2023-01-01` to `2024-07-01`. Therefore raw future covariates such as actual
2023 order count, traffic, promo usage, returns, or inventory cannot be used at
prediction time.

The compliant strategy is:

- aggregate every auxiliary source to daily history,
- convert each daily series into forecast-safe priors,
- use only seasonal means, calendar-year references, and long lags computed from
  rows available at each fold cutoff.

The implementation is in `src/aux_features.py`.

## 2. Daily aggregates built

Main aggregate families:

- Demand volume: `order_count`, `unique_customers`, `units`, `gross_sales`,
  `net_item_sales`, `payment_value`, `shipments`.
- Marketing and web: `avg_discount_rate`, `promo_item_share`, `active_promos`,
  `avg_promo_discount_value`, traffic sessions/page views, traffic source mix.
- Product and geography mix: category revenue shares, region revenue shares,
  product margin, unit price, mobile/desktop/payment mix.
- Ops and quality: return rate, refund amount, reviews/rating, shipping days,
  shipping fee, stockout and inventory health metrics.

Feature transforms for each selected daily aggregate:

- `month_mean`
- `month_dow_mean`
- `doy_mean`
- `week_mean`
- `lag548`
- `lag730`
- `year1`
- `year2`
- `roll30_548`
- `roll30_730`

## 3. Analysis findings

Strong raw correlations exist inside the historical period:

| Auxiliary signal | Corr with daily Revenue |
|---|---:|
| `gross_sales` | 1.000 |
| `net_item_sales` | 0.992 |
| `payment_value` | 0.992 |
| `unique_customers` | 0.937 |
| `order_count` | 0.936 |
| `units` | 0.918 |
| `shipments` | 0.815 |

These correlations are real but mostly not directly usable for future test rows
because the actual future values are unavailable. Once converted into
forecast-safe seasonal priors, much of the signal overlaps with existing
revenue seasonal features.

Most important auxiliary generated features from LightGBM split importance:

| Rank | Feature |
|---:|---|
| 1 | `aux_conversion_order_per_session_doy_mean` |
| 2 | `aux_net_item_sales_doy_mean` |
| 3 | `aux_desktop_share_doy_mean` |
| 4 | `aux_conversion_order_per_session_month_dow_mean` |
| 5 | `aux_unique_customers_year1` |
| 6 | `aux_paid_search_share_doy_mean` |
| 7 | `aux_region_share_central_year1` |
| 8 | `aux_shipments_doy_mean` |
| 9 | `aux_order_count_doy_mean` |
| 10 | `aux_shipping_fee_doy_mean` |

## 4. CV results

Mean over the same three walk-forward folds:

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| `ensemble_hist_gbr_baseline` | 594,615 | 805,960 | 0.745 | 0.225 |
| `gbr_log_target` | 606,892 | 833,222 | 0.727 | 0.228 |
| `hist_gbm_log_target` | 625,249 | 847,697 | 0.716 | 0.233 |
| `lightgbm_log_target` | 628,974 | 853,686 | 0.712 | 0.235 |
| `lightgbm_all_aux_log_target` | 636,496 | 867,219 | 0.704 | 0.239 |
| `lightgbm_top_aux_log_target` | 640,042 | 866,964 | 0.703 | 0.245 |
| `mlp_deep_learning` | 727,076 | 979,608 | 0.620 | 0.280 |
| `seasonal_naive_mean2y` | 705,507 | 963,072 | 0.634 | 0.271 |

Additional experiments:

- Top-N auxiliary feature selection (`10`, `20`, `40`, `80`, `160`) did not
  improve the production ensemble with a positive blend weight.
- A decomposition model using `orders` as an auxiliary target
  (`Revenue = predicted_order_count * predicted_average_order_value`) reached
  about `R2 = 0.728` after blending with the seasonal baseline, still below the
  current primary ensemble.

## 5. Decision

The auxiliary tables have useful explanatory value, but their test-window values
are unavailable. After converting them into forecast-safe priors, they did not
beat the current revenue-seasonality ensemble.

Final decision:

- Keep the current primary submission as `submissions/submission.csv`.
- Keep auxiliary models as benchmark submissions:
  `submissions/submission_lightgbm_top_aux_v5.csv` and
  `submissions/submission_lightgbm_all_aux_v5.csv`.
- Use the auxiliary analysis in the report to explain why the tables were
  considered, transformed safely, and not used as the winning production model.

Supporting artifacts:

- `outputs/aux_feature_importance_lgbm.csv`
- `outputs/aux_all_feature_cv.csv`
- `outputs/aux_top_feature_benchmark.csv`
- `outputs/aux_decomposition_benchmark.csv`
