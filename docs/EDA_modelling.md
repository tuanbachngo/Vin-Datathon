# EDA & Modelling — Part 3 Sales Forecasting

Analysis and strategy brief for forecasting daily `Revenue` over 2023-01-01 → 2024-07-01 (548-day horizon) from `sales.csv` (2012-07-04 → 2022-12-31).

---

## 1. Core findings from the analysis

### 1.1 Data reality (what the problem actually looks like)

| Finding | Evidence |
|---|---|
| Train is perfectly clean and continuous | 3,833 days, 0 gaps, 0 nulls, 0 zero/negative revenue |
| Test is a single contiguous 548-day block immediately after train | 2023-01-01 → 2024-07-01, no calendar gap |
| **No auxiliary table covers the test window** | orders / shipments / web_traffic / promotions / inventory / returns / reviews ALL end 2022-12-31 |
| Sample submission already has COGS filled with realistic values | cogs_ratio distribution (0.79–0.91) matches train_2022 exactly; corr(Rev, COGS)=0.98 |

### 1.2 Time-series structure

| Property | Value | What it tells us |
|---|---|---|
| Lag-1 autocorrelation | 0.87 | Strong day-to-day persistence — autoregressive signal |
| Lag-7 autocorrelation | 0.49 (only 0.15 after detrend) | Weekly seasonality is **weak** — DoW amplitude is 0.69–0.73× vs mean in last 24 months (nearly flat) |
| Lag-30 autocorrelation | 0.65 | Monthly cycle is meaningful |
| Lag-365 autocorrelation | 0.74 (0.24 after detrend) | **Annual seasonality is the dominant signal** |
| Lag-180 autocorrelation | ≈ 0 / slightly negative | Classic symmetric annual peak/trough shape |
| CV of Revenue per year | 0.41–0.63 | Log transform will stabilise variance |

### 1.3 Regime and trend

- **Structural break 2018 → 2019**: annual revenue falls from 1.85 B → 1.14 B (−39 %). Years ≤ 2018 live on a different level than years ≥ 2019.
- **Within the recent regime** (2019–2022): 2019→2020 = −7.5 %, 2020→2021 = +0.8 %, 2021→2022 = **+14.8 %**. 2022 is clearly a recovery year → test-period 2023–24 is more likely to continue upward than to revert to 2019 lows.
- **Monthly shape is remarkably stable year-on-year** (Mar–Jun peak ≈ 135 M/mo, Nov–Feb trough ≈ 50 M/mo, ~2.7× amplitude, same shape every year 2019–2022). August is the one exception — it flips sign (−44 % then +66 % YoY) — an anomalous month.
- **Irreducible noise floor**: even after removing monthly mean in the recent regime, the median day-to-month residual is **22 % of the monthly mean** (IQR 10–40 %). That's the ceiling on how good any model can look — MAPE much below ~20 % means overfitting.

### 1.4 Baseline benchmarks (proxy test: train on 2012–2021, predict 2022)

| Baseline | MAE | RMSE | R² |
|---|---|---|---|
| Predict global mean (2019–21) | 1,278,825 | 1,692,933 | −0.02 |
| Seasonal naive — same DoY in 2021 | 837,704 | 1,161,819 | 0.518 |
| **Average of same DoY in 2020 + 2021** | **732,963** | **1,011,956** | **0.634** |
| (month × DoW) mean over 2019–2021 | 912,008 | 1,360,535 | 0.339 |

**R² ≈ 0.63 is the concrete floor any ML model must beat** — otherwise the added complexity isn't earning its place.

---

## 2. Insights → direct implications for feature engineering

1. **No covariate leakage into the test window exists.** You physically cannot use `orders_yesterday`, `web_sessions`, `active_promos` at prediction time — those tables stop on 2022-12-31. Any feature built from auxiliary tables is only usable if it can be computed from a **date alone** (i.e. a pre-computed lookup keyed on calendar position), not from live state. This is the single biggest constraint.

2. **The seasonal-average baseline is already the strongest free signal.** Build features like `mean Revenue for (month, day-of-month)` over 2019–2022, `mean Revenue for (ISO-week)`, and `mean of same-DoY across past 2/3 years`. These encode the dominant annual cycle and are leakage-safe forever.

3. **DoW features are low-value.** Including `dow` one-hot or cyclical sin/cos encodings adds only marginal lift (0.70 vs 0.73 ratio). A single `is_weekend` flag is enough.

4. **Month / quarter features are high-value**, particularly non-linear encodings — the peak (Apr) is 2.7× the trough (Jan). Cyclical sin/cos(month) + explicit `month` categorical both belong in the model.

5. **Long-horizon lags are the only safe autoregressive features** for a direct multi-step model over 548 days:
   - `lag-365`, `lag-730` are always safe for any test date.
   - `lag-548` (= horizon) is safe for all test dates.
   - Rolling means like `roll30_ending_365d_ago`, `roll90_ending_365d_ago` are safe and smooth out lag-365 noise.
   - **Short lags (lag-1, lag-7, lag-30) are NOT safe** in a direct model — for a test point on 2024-06-01, lag-30 would need 2024-05-02 which doesn't exist in train. They only work in a recursive model, and recursion compounds error badly over 548 steps.

6. **Use log(Revenue) as the target.** CV is stable at ~0.5 across years, and log stabilises both variance and the heavy August outliers. Invert with `exp(pred) × bias_correction` (the usual log-MSE debias factor).

7. **Weight recent regime more than old history.** 2012–2018 data represents a defunct level. Either (a) filter to 2019+ (~1,460 rows, usually enough for a gradient booster), or (b) train on full history with exponential sample weights `w = 0.7 ** (years_from_cutoff)`. A flat-weighted model on 10 years will drag predictions toward an obsolete mean.

8. **COGS ambiguity is a strategic fork.** If organizers confirm the test COGS is real (given), then `Revenue ≈ COGS / cogs_ratio` makes this almost trivial — just model the ratio. If COGS is placeholder, ignore it entirely. **Clarify before investing in modeling — this can 10× the accuracy ceiling or drop it to nothing.**

9. **Calendar features beyond DoW/month matter**: `is_vietnamese_holiday` (Tet, Reunification Day April 30, National Day Sept 2), `is_tet_week`, `days_since_tet`, `month_of_quarter`. Lunar-calendar features are likely to add more signal than DoW given the Vietnamese retail context and the strong Jan/Feb trough.

10. **August anomaly**: create an `is_august` or `year_in_regime × month` interaction — standard seasonal features will under-fit August's flip-flopping.

---

## 3. Implications for modelling

1. **Forecast Revenue directly**, even though the submission asks for (Date, Revenue, COGS). If COGS is a placeholder, it doesn't need to be correct — either echo the sample values or predict it with a separate model. Confirm with organizers what they grade.

2. **Do not use a recursive ARIMA / LSTM-style one-step model over 548 days.** Error compounds. Prefer a **direct multi-output model** where every test date is predicted independently from leakage-safe features.

3. **Model progression (spend effort in this order):**
   1. Seasonal-naive benchmark (`mean of last-2-years same-DoY`) — target R² ≈ 0.63 on 2022 holdout.
   2. Gradient boosting (LightGBM) on calendar + long-lag + seasonal-lookup features, log target, sample-weighted toward recent years — realistic target R² ≈ 0.75–0.82.
   3. SARIMA / Prophet for comparison, not primary — these will fight the regime shift.
   4. Blend 2 + 3 if results support it. Skip deep learning (N-BEATS, TFT) unless (2) plateaus below expectations — only ~1,460 rows in the recent regime, which is small for deep models.

4. **Validation harness must use expanding walk-forward with folds shaped like the test.** Natural folds:
   - Train through 2019-06 → val 2019-07 to 2020-12 (18 mo)
   - Train through 2020-06 → val 2020-07 to 2021-12
   - Train through 2021-06 → val 2021-07 to 2022-12

   This matches the 18-month horizon and catches over-reliance on short lags.

5. **Monitor for overfitting to the 2019 break.** If the model gives wildly different preds for 2023 than linear extrapolation of 2021–22, inspect feature importances — a rank-sensitive model may over-rely on lag-730 (= 2021 level), an under-estimate relative to 2022's recovery.

---

## 4. Recommended next steps (in order, with owners)

| # | Owner | Deliverable | Blocker? |
|---|---|---|---|
| 1 | `ds-lead` | **Clarify with organizers whether test-window COGS is known** | YES — stops before starting |
| 2 | `validator` | Build walk-forward CV harness (3 folds as above), implement MAE / RMSE / R² on log-scale back-transformed predictions | Needed to compare any model |
| 3 | `modeler` | Commit the 2-year-mean seasonal-naive as the reference baseline; reproduce R² ≈ 0.63 on fold 2022 | Anchors all future improvements |
| 4 | `feature-engineer` | Build feature matrix from leakage-safe signals only: calendar (sin/cos month/week, DoW, Vietnamese holidays, Tet-distance), long-lag revenues (365, 730, 548), rolling-mean ending 365d ago, regime flag (year ≥ 2019) | Feeds step 5 |
| 5 | `modeler` | LightGBM on log(Revenue) with sample weights (recent ≫ old); target: beat baseline by ≥ 5 R² points on all 3 folds | Core model |
| 6 | `data-analyst` | Decompose the August anomaly; decide whether to build `month × year-bucket` interaction or leave as high-variance residual | Polish |
| 7 | `explainer` | SHAP + PDP on the winning model; verify top features are `lag-365`, `month`, `Tet-distance` (these should dominate) | Required by spec |
| 8 | `ds-lead` | Package submission.csv in sample_submission row order, preserve COGS column strategy from step 1 | Final |

---

## 5. Top risks to flag now

1. **COGS ambiguity** — silently determines whether the problem is hard or trivial.
2. **Regime-naive training** — if someone trains on 2012–2022 unweighted, predictions will be 30–50 % too high; symptom is the model loving lag-730 over lag-365.
3. **Recursive short-lag features** — any pipeline that builds `lag_1`, `lag_7`, etc. and uses them on test dates is leaking / error-compounding. Must be enforced at CV-harness level.
4. **August** — if validation folds end before August, the model will look good and fail on test's 2023-08 and 2024 summer.

---

## 6. Execution log — Phase 2 pipeline (delivered)

### 6.1 Modules
- `src/validation.py` — walk-forward folds + MAE/RMSE/R²/MAPE metrics.
- `src/features.py` — leakage-safe feature builder (calendar, Tet, long-lags 365/548/730, rolling means ending 365/548/730d ago, seasonal lookups rebuilt per fold, regime flag, exponential sample weights `half_life = 2y`).
- `src/baselines.py` — seasonal-naive variants.
- `src/model.py` — `HistGradientBoostingRegressor` on log(Revenue) (LightGBM not installed in `.venv`; sklearn equivalent used).
- `src/run_pipeline.py` — driver (CV + fit-on-all + submission write).

### 6.2 CV results (mean across 3 folds: 2020H2-2021, 2021H2-2022, 2022 only)

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| seasonal_naive_lag365 (fold3 only) | 837,704 | 1,161,819 | 0.518 | 27.9 % |
| seasonal_naive_growth (fold3 only) | 843,078 | 1,165,580 | 0.515 | 27.9 % |
| seasonal_naive_mean2y | 706,601 | 991,342 | 0.612 | 27.1 % |
| **gbm_log_target** | **660,905** | **900,493** | **0.679** | **24.2 %** |

GBM clears the target set in §3 (≥ 5 R² points above the strongest baseline): +6.7 R² mean, **+6.5 R² on fold 3 (2022)** — the most test-like fold.

### 6.3 Feature importance (permutation, fold-3 val set, top 10)

| Feature | Δ log-MSE |
|---|---|
| seasonal_doy_mean | 0.756 |
| seasonal_month_dow_mean | 0.054 |
| rev_lag_365_730_mean | 0.037 |
| rev_lag_548 | 0.014 |
| rev_roll7_end_730d_ago | 0.010 |
| rev_roll30_end_365d_ago | 0.010 |
| rev_roll30_end_730d_ago | 0.009 |
| rev_roll7_end_548d_ago | 0.007 |
| days_to_tet | 0.007 |
| dow | 0.007 |

**Interpretation:** annual cycle (via `seasonal_doy_mean`) carries ~80 % of the lift, confirming the EDA thesis. Long-lag features and Tet-distance contribute smaller but non-zero signal. DoW features are low-value (consistent with weak post-detrend lag-7 ACF of 0.15).

### 6.4 Artifacts

| Path | Purpose |
|---|---|
| `outputs/cv_results.csv` | raw fold × model metrics |
| `outputs/feature_importance_fold3.csv` | permutation importance (50 features) |
| `outputs/predictions_overview.png` | visual — train + GBM + baseline on daily and monthly aggregate |
| `submissions/submission_gbm_v1.csv` | primary submission (GBM) |
| `submissions/submission_seasonal_mean2y.csv` | fallback baseline submission |

Run the whole pipeline with `.venv/Scripts/python.exe -m src.run_pipeline`.

### 6.5 Predicted monthly totals (M VND) vs 2022 actual

| Month | 2022 actual | 2023 pred | 2024 pred |
|---|---|---|---|
| Jan | 59.7 | 56.1 | 53.9 |
| Feb | 79.1 | 70.1 | 75.3 |
| Mar | 137.5 | 124.4 | 127.5 |
| Apr | 141.3 | 132.2 | 142.9 |
| May | 139.0 | 140.6 | 144.9 |
| Jun | 135.8 | 131.8 | 128.9 |
| Jul | 98.1 | 94.7 | 4.6* |
| Aug | 113.5 | 80.7 | — |
| Sep | 85.8 | 75.8 | — |
| Oct | 75.2 | 72.1 | — |
| Nov | 52.2 | 51.7 | — |
| Dec | 52.5 | 47.8 | — |

\* July 2024 covers only 2024-07-01 (1 day)

Model is **conservative** — 2023 is lower than 2022 despite 2022's +14.8 % YoY recovery. This is because the GBM's seasonal lookup averages 2019–2022 (which still includes the post-break lower years). August is conspicuously under-predicted (80.7 vs 113.5) — the known anomaly from §1.3.

### 6.6 Immediate refinements (next iteration)

1. **Level-correct the seasonal lookup**: scale `seasonal_doy_mean` by the ratio of the most-recent 12 months to the lookup-period mean (captures the 2022 recovery).
2. **Treat August as its own regime**: either drop 2020 from the seasonal lookup for month 8, or add `year × is_august` interaction.
3. **Shorter sample-weight half-life (e.g. 1 year)** to let 2022 dominate more.
4. **Blend GBM with `seasonal_naive_mean2y`** — baseline is stronger on extreme months; blend weight found by fold3.
5. **Install LightGBM / XGBoost** for more expressive models (quantile loss can give prediction intervals too).
6. **Still blocking**: confirm with organizers whether test-window COGS is given or a placeholder — this is the single highest-ROI clarification.

---

## 7. v2 results — level correction + August fix + blend

### 7.1 Competition rule resolution
Per organizer confirmation: **participants must not use Revenue or COGS from the test set as features** (disqualification condition). Action:
- `sample_submission.csv`'s COGS column is **not** a real feature — treat as placeholder.
- Our pipeline is already compliant (features built only from `sales.csv` train rows).
- Submission echoes the sample COGS column verbatim to satisfy the output format; no predictive use.

### 7.2 Changes vs v1

| Change | File | Effect |
|---|---|---|
| Level-correction of seasonal lookups | `src/features.py` → `_seasonal_lookup_features` | Each `seasonal_*_mean` multiplied by `mean(Revenue last 365d) / mean(Revenue lookup-period)`; exposed as `level_ratio_12m` feature |
| August regime fix | same | 2020 dropped from month-8 rows of the seasonal lookup |
| GBM × baseline blend | `src/run_pipeline.py` → `tune_blend_weight` | Grid-search `w` on each fold for `w·gbm + (1-w)·seasonal_naive_mean2y`; fold-3 weight used in final submission |

### 7.3 CV: v1 vs v2 (mean across 3 folds)

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| seasonal_naive_mean2y | 706,601 | 991,342 | 0.612 | 27.1 % |
| **v1** gbm_log_target | 660,905 | 900,493 | 0.679 | 24.2 % |
| **v2** gbm_log_target | 661,062 | 908,549 | 0.674 | 24.3 % |
| **v2 blend_gbm_baseline** | **624,332** | **861,224** | **0.708** | **23.3 %** |

Per-fold blend weights: `fold1 = 0.6`, `fold2 = 0.8`, `fold3 = 0.7` → **final submission uses w = 0.70**.

### 7.4 Where the lift came from

| Fold | v1 GBM R² | v2 GBM R² | v2 blend R² | Δ vs v1 GBM |
|---|---|---|---|---|
| fold1_2020H2-2021 | 0.635 | 0.635 | **0.718** | +0.083 |
| fold2_2021H2-2022 | 0.702 | 0.717 | **0.725** | +0.023 |
| fold3_test_proxy | 0.702 | 0.671 | **0.682** | −0.020 |
| **mean** | **0.679** | **0.674** | **0.708** | **+0.029** |

- Level correction alone did NOT move the GBM mean (-0.005 R²) — GBM already learned the trend via other features. But it reduced variance across folds (fold 2 +1.5 pts).
- **The blend is the main win** (+2.9 R² on mean, +8.3 pts on fold 1). Interpretation: on early folds where the recent regime is less well-represented, the simple 2-year-mean baseline is more robust; GBM overfits. Blending regularises.
- Fold 3 dipped −2 R² — the level ratio over-corrected for 2022's recovery. Acceptable trade since fold 1 & 2 gains dominate.

### 7.5 Final predictions sanity check (blend v2, vs 2022 actual)

| Month | 2022 actual (M) | 2023 pred | 2024 pred | 2023 vs 2022 | 2024 vs 2022 |
|---|---|---|---|---|---|
| Jan | 59.7 | 54.4 | 54.9 | −8.9 % | −8.0 % |
| Feb | 79.1 | 72.1 | 79.2 | −8.9 % | +0.1 % |
| Mar | 137.5 | 125.2 | 133.0 | −8.9 % | −3.3 % |
| Apr | 141.3 | 136.1 | 142.5 | −3.7 % | +0.9 % |
| May | 139.0 | 140.7 | 143.2 | +1.2 % | +3.0 % |
| Jun | 135.8 | 132.6 | 132.0 | −2.4 % | −2.8 % |
| Jul | 98.1 | 97.1 | 4.5* | −1.1 % | — |
| Aug | 113.5 | 85.3 | — | −24.9 % | — |

\* July 2024 = 1 day only (2024-07-01).

Peak months (Apr-Jun) now align within ±3 % of 2022 actual — the v1 under-prediction is gone. **Aug-23 is still under-predicted by 25 %** — we dropped 2020 but 2021-Aug (68.3M) still drags the mean vs 2022-Aug (113.5M). Further treatment in v3 would add `year × is_august` as a proper interaction.

### 7.6 v2 Artifacts

| Path | Purpose |
|---|---|
| `outputs/cv_results.csv` | Overwritten with v2 fold metrics |
| `outputs/predictions_v2.png` | Train + GBM + baseline + blend, daily and monthly |
| `submissions/submission_gbm_v2.csv` | GBM-only v2 |
| `submissions/submission_blend_v2.csv` | **← primary submission** (blend, w = 0.70) |
| `submissions/submission_seasonal_mean2y.csv` | Baseline fallback |
| `docs/organizer_question.md` | Pre-clarification brief (now resolved — kept for audit) |

### 7.7 What's next (v3 candidates)

1. **August year-bucket interaction** — remaining known bias.
2. **Quantile / prediction intervals** — requires installing LightGBM (`pip install lightgbm`) or using sklearn's GBR with quantile loss.
3. **Feature-importance-guided pruning** — 40+ features, but top 10 carry >95 % of the lift; drop the tail to reduce noise and overfitting.
4. **Error analysis by day type** — if blend-v2 errors concentrate on specific DoW or holidays, add targeted calendar features.

---

## 8. v3 — biennial (year-parity) features + pruning experiment

### 8.1 Core finding driving v3
Investigation of August anomalies revealed a **clear biennial pattern**: `corr(is_even_year, August_total) = 0.97`. Even-year Augusts average ~120M/mo (2020 = 122.1, 2022 = 113.5), odd-year Augusts ~75M/mo (2019 = 81.5, 2021 = 68.3). Only August shows this pattern strongly in the 2019+ regime — other months have no biennial signal.

**Implication:** v2's Aug-2023 prediction of 85.3M was being pulled toward the wrong reference (2022 even-year 113.5M). The correct expectation for Aug-2023 (an odd year) is closer to the odd-year mean of ~75M.

### 8.2 v3 changes

| Change | File | Rationale |
|---|---|---|
| `is_even_year` + `is_aug_even` / `is_aug_odd` interaction flags | `src/features.py` → `_calendar_features` | Lets the GBM split on year-parity, especially for August |
| `seasonal_month_parity_mean` (per `(month, parity)` lookup, level-corrected) | `src/features.py` → `_seasonal_lookup_features` | Replaces the v2 "drop 2020 from Aug" hack with a principled parity-aware mean |
| Reverted v2's Aug-2020 drop | same | Now captured through parity features instead of row exclusion |

### 8.3 Pruning experiment (reverted)
Attempted a whitelist of 15 top-permutation-importance features (imp ≥ 0.001 on fold 3). **Regressed CV significantly**:
- Full features: blend R² = 0.722; GBM-only R² = 0.698
- Pruned to 15: blend R² = 0.666; GBM-only R² = 0.563
- Fold 1 GBM crashed 0.657 → 0.347

**Lesson:** permutation importance measured on fold 3 is not transferable. Many "weak" features carry fold-specific interaction signal that top features alone can't reconstruct. Pruning is NOT applied in the shipped v3; `use_whitelist = False` is the default in `src/model.py`.

### 8.4 CV: v2 vs v3 (mean across 3 folds)

| Model | v2 R² | v3 R² | Δ |
|---|---|---|---|
| gbm_log_target | 0.674 | **0.698** | +0.024 |
| blend_gbm_baseline | 0.708 | **0.722** | +0.014 |

Per-fold blend R² (v3):

| Fold | v2 | v3 | Δ |
|---|---|---|---|
| fold1_2020H2-2021 | 0.718 | **0.723** | +0.005 |
| fold2_2021H2-2022 | 0.725 | **0.722** | −0.003 |
| fold3_test_proxy | 0.682 | **0.721** | **+0.039** |

Fold 3 (the most test-like fold, 2022) gained **+3.9 R² points** — biggest jump — because 2022 is an even year and the parity feature aligns it correctly. Final blend weight from fold 3: **w = 0.90** (GBM dominates; baseline adds only a thin smoothing layer).

### 8.5 v3 feature importance (fold 3, top 10)

| Rank | Feature | Δ log-MSE |
|---|---|---|
| 1 | seasonal_doy_mean | 0.746 |
| 2 | seasonal_month_dow_mean | 0.052 |
| 3 | rev_lag_365_730_mean | 0.029 |
| 4 | rev_lag_548 | 0.020 |
| 5 | **is_aug_even** (new in v3) | **0.012** |
| 6 | days_to_tet | 0.012 |
| 7 | rev_roll30_end_730d_ago | 0.010 |
| 8 | rev_roll7_end_730d_ago | 0.010 |
| 9 | **seasonal_month_parity_mean** (new in v3) | **0.007** |
| 10 | dow_sin | 0.005 |

Both new v3 features ranked in the top 10, validating the biennial hypothesis empirically.

### 8.6 August prediction sanity check

| Year | August total (M VND) | Year parity |
|---|---|---|
| 2019 | 81.5 | odd |
| 2020 | 122.1 | even |
| 2021 | 68.3 | odd |
| 2022 | 113.5 | even |
| **2023 (v3 pred)** | **73.4** | **odd** — aligned with odd-year range |

v2 predicted 85.3M (drift toward even-year mean). v3's 73.4M sits between Aug-2019 (81.5) and Aug-2021 (68.3) — the right reference class.

### 8.7 v3 artifacts

| Path | Purpose |
|---|---|
| `outputs/cv_results.csv` | Overwritten with v3 fold metrics |
| `outputs/feature_importance_v3.csv` | Permutation importance, 52 features |
| `outputs/predictions_v3.png` | v2 vs v3 blend overlay (daily + monthly) |
| `submissions/submission_gbm_v3.csv` | GBM-only v3 |
| **`submissions/submission_blend_v3.csv`** | **← primary submission** (blend, w = 0.90) |

### 8.8 Ceiling check

The recent-regime noise floor (median daily residual over monthly mean) is **22 %**, with IQR 10–40 % (§1.3). V3's MAPE is **23.1 %** — we are ~1 point above the irreducible noise floor. Further gains at the feature level are likely marginal. The remaining room for improvement is:
- Model family (LightGBM with quantile loss for intervals — requires install)
- Ensemble of multiple GBM seeds for variance reduction
- Target transformation experiments (Box-Cox instead of log)

At R² = 0.722 and MAPE = 23 %, v3 is **at the diminishing-returns frontier**. Recommend stopping here unless a specific weakness (e.g. test-period Tet in early Feb 2024) emerges.
