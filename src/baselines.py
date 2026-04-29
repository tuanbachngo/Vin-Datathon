from __future__ import annotations
import numpy as np
import pandas as pd


def _safe_mean(*arrays: np.ndarray) -> np.ndarray:
    stacked = np.vstack(arrays).astype(float)
    valid = np.sum(~np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    out = np.full(total.shape, np.nan, dtype=float)
    np.divide(total, valid, out=out, where=valid > 0)
    return out


def _calendar_year_back(
    target_dates: pd.Series, series: pd.Series, years: int
) -> np.ndarray:
    shifted = pd.to_datetime(target_dates) - pd.DateOffset(years=years)
    return series.reindex(shifted).values


def seasonal_naive_last_year(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    """Predict Revenue[t] from the same calendar date last year."""
    s = sales[sales.Date <= as_of].set_index("Date").Revenue
    return _calendar_year_back(target_dates, s, years=1)


def seasonal_naive_mean_2y(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    """Predict Revenue[t] from the same calendar date 1y and 2y back.

    For late-horizon dates where the 1-year-back value is unavailable
    (e.g. 2024 dates when train ends at 2022-12-31), this naturally falls back
    to the 2-year-back value only.
    """
    s = sales[sales.Date <= as_of].set_index("Date").Revenue
    a = _calendar_year_back(target_dates, s, years=1)
    b = _calendar_year_back(target_dates, s, years=2)
    return _safe_mean(a, b)


def seasonal_naive_growth_adjusted(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    """Same-calendar-date last year scaled by recent/prior 12-month level.

    Level ratio = mean(Revenue[as_of-365:as_of]) / mean(Revenue[as_of-730:as_of-365]).
    """
    s = sales[sales.Date <= as_of].set_index("Date").sort_index().Revenue
    recent = s[s.index > as_of - pd.Timedelta(days=365)].mean()
    prior = s[(s.index > as_of - pd.Timedelta(days=730))
              & (s.index <= as_of - pd.Timedelta(days=365))].mean()
    ratio = recent / prior if prior > 0 else 1.0
    preds = _calendar_year_back(target_dates, s, years=1) * ratio
    return preds


def seasonal_lookup_level_adjusted(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    hist = sales[sales.Date <= as_of].copy()
    if hist.empty:
        return np.full(len(target_dates), np.nan, dtype=float)

    hist["month"] = hist.Date.dt.month
    hist["dow"] = hist.Date.dt.dayofweek
    recent_mask = hist.Date.dt.year >= 2019
    base = hist.loc[recent_mask].copy() if recent_mask.sum() > 300 else hist.copy()

    month_mean = base.groupby("month").Revenue.mean()
    month_dow_mean = base.groupby(["month", "dow"]).Revenue.mean()

    last_year_mean = hist.loc[hist.Date > (as_of - pd.Timedelta(days=365)), "Revenue"].mean()
    lookup_mean = base["Revenue"].mean()
    level_ratio = (last_year_mean / lookup_mean) if lookup_mean > 0 else 1.0

    dates = pd.to_datetime(target_dates)
    out = pd.DataFrame({"Date": dates})
    out["month"] = out.Date.dt.month
    out["dow"] = out.Date.dt.dayofweek
    out["prediction"] = out.apply(
        lambda row: month_dow_mean.get((row.month, row.dow), np.nan), axis=1
    )
    out["prediction"] = out["prediction"].fillna(out["month"].map(month_mean))
    out["prediction"] = out["prediction"] * level_ratio
    fallback = float(last_year_mean) if np.isfinite(last_year_mean) else float(hist["Revenue"].mean())
    out["prediction"] = out["prediction"].fillna(fallback)
    return out["prediction"].to_numpy(dtype=float)


def seasonal_residual_baseline(
    target_dates: pd.Series, sales: pd.DataFrame, as_of: pd.Timestamp
) -> np.ndarray:
    mean_2y = seasonal_naive_mean_2y(target_dates, sales, as_of)
    lookup = seasonal_lookup_level_adjusted(target_dates, sales, as_of)

    preds = np.asarray(mean_2y, dtype=float)
    preds = np.where(np.isfinite(preds), preds, lookup)
    preds = np.where(np.isfinite(preds) & (preds > 0), preds, np.nan)
    if np.isnan(preds).any():
        safe_fill = np.nanmedian(lookup) if np.isfinite(lookup).any() else 1.0
        preds = np.where(np.isnan(preds), safe_fill, preds)
    return preds


_seasonal_lookup_level_adjusted = seasonal_lookup_level_adjusted
