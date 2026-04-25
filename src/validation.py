"""Walk-forward CV harness for the 548-day revenue forecast."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def mask_train(self, dates: pd.Series) -> pd.Series:
        return dates <= self.train_end

    def mask_val(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.val_start) & (dates <= self.val_end)


def default_folds() -> list[Fold]:
    """Three walk-forward folds shaped like the real test (18-month horizon).

    Each val window spans ~18 months ending on Dec-31 so August anomalies
    are always inside validation.
    """
    return [
        Fold("fold1_2020H2-2021", pd.Timestamp("2020-06-30"),
             pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
        Fold("fold2_2021H2-2022", pd.Timestamp("2021-06-30"),
             pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        Fold("fold3_test_proxy", pd.Timestamp("2021-12-31"),
             pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    ]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs(err) / np.maximum(y_true, 1.0)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def summarize_folds(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    mean_row = df.select_dtypes("number").mean().to_dict()
    mean_row["fold"] = "MEAN"
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
