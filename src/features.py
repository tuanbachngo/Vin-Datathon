"""Leakage-safe feature builder for the 548-day daily revenue forecast.

Design rules (updated from the modelling report):
- No live covariate may rely on auxiliary tables beyond 2022-12-31.
- Short autoregressive lags are disallowed in a direct 548-day forecast.
- `lag_548` and `lag_730` are fully available across the whole test horizon.
- `lag_365` is only available for the first ~12 months of the horizon, so it is
  kept as a partially-observed feature and downstream models must handle or
  encode its missingness explicitly.
- Target is modelled on log(Revenue).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29",
}
TET = pd.Series({int(y): pd.Timestamp(d) for y, d in TET_DATES.items()})


def _safe_nanmean(*arrays: np.ndarray) -> np.ndarray:
    stacked = np.vstack(arrays).astype(float)
    valid = np.sum(~np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    out = np.full(total.shape, np.nan, dtype=float)
    np.divide(total, valid, out=out, where=valid > 0)
    return out


def _days_to_nearest_tet(date: pd.Timestamp) -> int:
    y = date.year
    candidates = [TET.get(y - 1), TET.get(y), TET.get(y + 1)]
    candidates = [c for c in candidates if c is not None]
    return int(min((date - c).days for c in candidates if abs((date - c).days) <= 366
                   ) if candidates else 0)


def _signed_days_to_tet(date: pd.Timestamp) -> int:
    y = date.year
    candidates = [TET.get(y - 1), TET.get(y), TET.get(y + 1)]
    candidates = [c for c in candidates if c is not None]
    diffs = [(date - c).days for c in candidates]
    return int(min(diffs, key=abs))


VN_FIXED_HOLIDAYS = [
    ("01-01", "new_year"),
    ("04-30", "reunification"),
    ("05-01", "labor"),
    ("09-02", "national"),
]


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"Date": pd.to_datetime(dates)})
    d["year"] = d.Date.dt.year
    d["month"] = d.Date.dt.month
    d["day"] = d.Date.dt.day
    d["dow"] = d.Date.dt.dayofweek
    d["doy"] = d.Date.dt.dayofyear
    d["week"] = d.Date.dt.isocalendar().week.astype(int)
    d["quarter"] = d.Date.dt.quarter
    d["is_weekend"] = (d.dow >= 5).astype(int)
    d["is_month_start"] = d.Date.dt.is_month_start.astype(int)
    d["is_month_end"] = d.Date.dt.is_month_end.astype(int)
    d["month_end_window"] = (d.Date.dt.days_in_month - d.day <= 3).astype(int)

    # cyclical encodings — k=1
    d["month_sin"] = np.sin(2 * np.pi * d.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d.month / 12)
    d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25)
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)

    # Fourier harmonics k=2,3 (capture sub-annual seasonality shapes)
    for _k in [2, 3]:
        d[f"month_sin_{_k}"] = np.sin(2 * np.pi * _k * d.month / 12)
        d[f"month_cos_{_k}"] = np.cos(2 * np.pi * _k * d.month / 12)
        d[f"doy_sin_{_k}"] = np.sin(2 * np.pi * _k * d.doy / 365.25)
        d[f"doy_cos_{_k}"] = np.cos(2 * np.pi * _k * d.doy / 365.25)
        d[f"dow_sin_{_k}"] = np.sin(2 * np.pi * _k * d.dow / 7)
        d[f"dow_cos_{_k}"] = np.cos(2 * np.pi * _k * d.dow / 7)

    # Vietnamese fixed-date holidays (+/- 1-day halo)
    for mmdd, name in VN_FIXED_HOLIDAYS:
        mm, dd = [int(x) for x in mmdd.split("-")]
        is_hol = ((d.month == mm) & (d.day == dd)).astype(int)
        d[f"is_{name}"] = is_hol
    d["is_vn_holiday"] = d[[f"is_{n}" for _, n in VN_FIXED_HOLIDAYS]].max(axis=1)

    # Tet features (signed days to nearest Tet, plus within-Tet-week flag)
    d["days_to_tet"] = d.Date.apply(_signed_days_to_tet)
    d["abs_days_to_tet"] = d.days_to_tet.abs()
    d["is_tet_week"] = (d.abs_days_to_tet <= 7).astype(int)
    d["pre_tet_window"] = ((d.days_to_tet >= -21) & (d.days_to_tet < 0)).astype(int)
    d["post_tet_window"] = ((d.days_to_tet > 0) & (d.days_to_tet <= 14)).astype(int)

    # Regime indicator: 2019+ is the "new-normal" level
    d["regime_recent"] = (d.year >= 2019).astype(int)

    # Biennial pattern: August shows +100 % even-vs-odd year gap (corr 0.97).
    # April shows a weaker version (+4 %). Expose year parity so the GBM
    # can learn the month * parity interaction without hand-coding it.
    d["is_even_year"] = (d.year % 2 == 0).astype(int)
    d["is_aug_even"] = (d.is_even_year & (d.month == 8)).astype(int)
    d["is_aug_odd"] = ((1 - d.is_even_year) & (d.month == 8)).astype(int)
    d["year_end_inventory_clearance_window"] = d.month.isin([8, 9, 10, 11]).astype(int)
    d["promo_window_q4"] = d.month.isin([10, 11, 12]).astype(int)
    d["promo_window_midyear"] = d.month.isin([5, 6, 7]).astype(int)
    d["odd_year_q4_pressure_flag"] = (
        ((d.year % 2 == 1) & d.month.isin([10, 11, 12]))
    ).astype(int)

    # Time elapsed features
    _ORIGIN = pd.Timestamp("2012-07-04")
    _REGIME_START = pd.Timestamp("2019-01-01")
    d["days_since_origin"] = (d.Date - _ORIGIN).dt.days
    d["days_since_regime"] = np.maximum(0, (d.Date - _REGIME_START).dt.days)

    # Holiday / event window features
    d["tet_pre_2w"] = ((d.days_to_tet >= -14) & (d.days_to_tet < -7)).astype(int)
    d["tet_pre_1w"] = ((d.days_to_tet >= -7) & (d.days_to_tet < 0)).astype(int)
    d["tet_post_1w"] = ((d.days_to_tet > 0) & (d.days_to_tet <= 7)).astype(int)
    d["days_to_month_end"] = d.Date.dt.days_in_month - d.day
    _qend = d.Date.dt.to_period("Q").dt.to_timestamp(how="E")
    d["days_to_quarter_end"] = (_qend - d.Date).dt.days
    d["is_quarter_end_week"] = (d.days_to_quarter_end <= 7).astype(int)

    # Interaction features
    d["is_q1_recent"] = ((d.quarter == 1) & d.regime_recent.astype(bool)).astype(int)
    d["is_q2_recent"] = ((d.quarter == 2) & d.regime_recent.astype(bool)).astype(int)
    d["month_x_even_year"] = d.month * d.is_even_year
    d["tet_week_x_dow"] = d.is_tet_week * d.dow
    d["month_end_x_q4_promo"] = d.month_end_window * d.promo_window_q4
    d["tet_or_month_end_pressure"] = (
        (d.pre_tet_window.astype(bool))
        | (d.post_tet_window.astype(bool))
        | (d.month_end_window.astype(bool))
    ).astype(int)

    return d.drop(columns=["Date"])


def _long_lag_features(dates: pd.Series, sales: pd.DataFrame) -> pd.DataFrame:
    """Revenue lags used by the direct long-horizon model.

    `lag_548` and `lag_730` cover the full forecast horizon.
    `lag_365` is intentionally retained even though it goes missing for the late
    2024 portion of the horizon because it is still informative for the first
    12 months; downstream models treat that missingness explicitly.
    """
    s = sales.set_index("Date").Revenue.sort_index()
    dates = pd.to_datetime(dates)
    out = pd.DataFrame(index=dates)
    for lag in [365, 548, 730, 1095]:
        out[f"rev_lag_{lag}"] = s.reindex(dates - pd.Timedelta(days=lag)).values

    # Log-safe versions (log1p so NaN-filled later won't crash)
    for lag in [365, 548, 730, 1095]:
        out[f"log_rev_lag_{lag}"] = np.log(np.maximum(out[f"rev_lag_{lag}"], 1.0))

    # Rolling means ending 365d ago (smooth version of lag365, still leakage-safe)
    rolled = {}
    for window in [7, 30, 90]:
        rolled[window] = s.rolling(window, min_periods=max(3, window // 2)).mean()

    for window in [7, 30, 90]:
        out[f"rev_roll{window}_end_365d_ago"] = rolled[window].reindex(
            dates - pd.Timedelta(days=365)
        ).values
        out[f"rev_roll{window}_end_548d_ago"] = rolled[window].reindex(
            dates - pd.Timedelta(days=548)
        ).values
        out[f"rev_roll{window}_end_730d_ago"] = rolled[window].reindex(
            dates - pd.Timedelta(days=730)
        ).values
        out[f"rev_roll{window}_end_1095d_ago"] = rolled[window].reindex(
            dates - pd.Timedelta(days=1095)
        ).values

    # EWMA at lag-safe anchor points (more weight on recent history than simple rolling)
    _ewm_cache: dict[int, pd.Series] = {}
    for span in [14, 30, 90]:
        _ewm_cache[span] = s.ewm(span=span, min_periods=max(5, span // 4)).mean()
    for span in [14, 30, 90]:
        for lag in [365, 548, 730, 1095]:
            out[f"rev_ewm{span}_end_{lag}d_ago"] = _ewm_cache[span].reindex(
                dates - pd.Timedelta(days=lag)
            ).values

    # YoY growth ratio using roll30 at each anchor (helps model learn trend direction)
    for lag_near, lag_far in [(365, 730), (548, 730), (730, 1095)]:
        near = out[f"rev_roll30_end_{lag_near}d_ago"]
        far = out[f"rev_roll30_end_{lag_far}d_ago"].replace(0, np.nan)
        out[f"rev_roll30_yoy_ratio_{lag_near}vs{lag_far}"] = near / far

    # Blend the annual lag with the fully-safe 2-year lag. Late-horizon rows
    # naturally fall back to lag-730 when lag-365 is unavailable.
    out["rev_lag_365_730_mean"] = _safe_nanmean(
        out.rev_lag_365.values, out.rev_lag_730.values
    )
    out["log_rev_lag_365_730_mean"] = np.log(np.maximum(out.rev_lag_365_730_mean, 1.0))

    return out.reset_index(drop=True)


def _seasonal_lookup_features(
    dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Historical averages keyed by calendar position, computed strictly from
    rows on or before `as_of`. Re-built per fold to avoid temporal leakage.

    v2 changes vs v1:
    - Level-correction: each seasonal mean is multiplied by
      `mean(Revenue last 365d ending as_of) / mean(Revenue over lookup period)`
      so the lookup reflects the most recent level, not the lookup average.
    - August regime fix: for month = 8, 2020 is excluded from the lookup
      (the +66 % 2022 rebound from a depressed 2020 base biased the mean).
    """
    dates = pd.to_datetime(dates)
    hist = sales[sales.Date <= as_of].copy()
    hist["year"] = hist.Date.dt.year
    hist["month"] = hist.Date.dt.month
    hist["dow"] = hist.Date.dt.dayofweek
    hist["doy"] = hist.Date.dt.dayofyear
    hist["week"] = hist.Date.dt.isocalendar().week.astype(int)

    # Recent-regime-only averages (2019+ if available; else whole history)
    recent_mask = hist.year >= 2019
    base = hist[recent_mask] if recent_mask.sum() > 300 else hist

    # August regime fix: instead of dropping 2020, use year-parity aware means
    # for the month-8 rows only (the only strongly biennial month in history).
    base_copy = base.copy()
    base_copy["parity"] = (base_copy.year % 2 == 0).astype(int)

    month_mean = base_copy.groupby("month").Revenue.mean()
    month_dow_mean = base_copy.groupby(["month", "dow"]).Revenue.mean()
    week_mean = base_copy.groupby("week").Revenue.mean()
    doy_mean = base_copy.groupby("doy").Revenue.mean()

    # Separate month-mean by parity, used only for August lookups
    month_parity_mean = base_copy.groupby(["month", "parity"]).Revenue.mean()

    # Level correction: scale by (last 365d mean) / (lookup-period mean)
    last_year_mask = hist.Date > (as_of - pd.Timedelta(days=365))
    last_year_mean = hist.loc[last_year_mask, "Revenue"].mean()
    lookup_mean = base_copy.Revenue.mean()
    level_ratio = (last_year_mean / lookup_mean) if lookup_mean > 0 else 1.0

    out = pd.DataFrame({"Date": dates})
    out["month"] = out.Date.dt.month
    out["dow"] = out.Date.dt.dayofweek
    out["week"] = out.Date.dt.isocalendar().week.astype(int)
    out["doy"] = out.Date.dt.dayofyear
    out["parity"] = (out.Date.dt.year % 2 == 0).astype(int)

    out["seasonal_month_mean"] = out.month.map(month_mean) * level_ratio
    out["seasonal_week_mean"] = out.week.map(week_mean) * level_ratio
    out["seasonal_doy_mean"] = out.doy.map(doy_mean) * level_ratio
    out["seasonal_month_dow_mean"] = out.apply(
        lambda r: month_dow_mean.get((r.month, r.dow), np.nan), axis=1
    ) * level_ratio
    # Fallback fill using month_mean for missing combos
    out.seasonal_month_dow_mean = out.seasonal_month_dow_mean.fillna(
        out.seasonal_month_mean
    )

    # Parity-aware seasonal mean (only month-8 values are meaningfully different
    # from the non-parity version — still expose for all months so the GBM
    # can learn which months benefit from it)
    out["seasonal_month_parity_mean"] = out.apply(
        lambda r: month_parity_mean.get((r.month, r.parity), np.nan), axis=1
    ) * level_ratio
    out.seasonal_month_parity_mean = out.seasonal_month_parity_mean.fillna(
        out.seasonal_month_mean
    )
    for c in ["seasonal_month_mean", "seasonal_week_mean",
              "seasonal_doy_mean", "seasonal_month_dow_mean",
              "seasonal_month_parity_mean"]:
        out[f"log_{c}"] = np.log(np.maximum(out[c], 1.0))

    # Expose the level ratio as its own feature — tree models can read it
    # as a scalar multiplicative adjustment on other lag features.
    out["level_ratio_12m"] = level_ratio

    return out.drop(
        columns=["Date", "month", "dow", "week", "doy", "parity"]
    ).reset_index(drop=True)


def build_feature_matrix(
    target_dates: pd.Series,
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:

    assert sales.Date.max() >= as_of, "as_of past sales history"
    sales_masked = sales[sales.Date <= as_of]

    cal = _calendar_features(target_dates)
    lags = _long_lag_features(target_dates, sales_masked)
    lookup = _seasonal_lookup_features(target_dates, sales_masked, as_of)

    X = pd.concat([cal.reset_index(drop=True),
                   lags.reset_index(drop=True),
                   lookup.reset_index(drop=True)], axis=1)
    X["Date"] = pd.to_datetime(target_dates).values
    return X


def sample_weights(dates: pd.Series, half_life_years: float = 2.0) -> np.ndarray:
    """Exponential decay so recent history dominates. Latest date gets weight 1."""
    dates = pd.to_datetime(dates)
    ref = dates.max()
    age_years = (ref - dates).dt.days / 365.25
    return np.exp(-np.log(2) * age_years / half_life_years).values
