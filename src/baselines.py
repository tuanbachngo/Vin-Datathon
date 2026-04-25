"""Baseline forecasters."""
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
