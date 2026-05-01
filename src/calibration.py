from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:  # Package import when used from notebooks: from src.calibration import ...
    from .validation import metrics
except ImportError:  # Direct script-style import when src/ is on sys.path.
    from validation import metrics


@dataclass
class RevenueCalibrator:
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def predict(self, raw_predictions: np.ndarray | pd.Series) -> np.ndarray:
        raw_predictions = np.asarray(raw_predictions, dtype=float)
        raw_log = np.log(np.clip(raw_predictions, 1e-6, None))
        calibrated_log = np.interp(
            raw_log,
            self.x_thresholds,
            self.y_thresholds,
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )
        return np.exp(calibrated_log)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prediction_log_threshold": self.x_thresholds,
                "target_log_threshold": self.y_thresholds,
            }
        )


@dataclass
class RevenueCalibrationResult:
    calibrator: RevenueCalibrator
    grouped_frame: pd.DataFrame
    calibrated_oof: pd.DataFrame
    summary: dict[str, float]


def fit_revenue_calibrator(oof_predictions: pd.DataFrame) -> RevenueCalibrationResult:
    required = {"Date", "actual_revenue", "prediction_raw"}
    missing = required.difference(oof_predictions.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")

    grouped = (
        oof_predictions.groupby("Date", as_index=False)
        .agg(
            actual_revenue=("actual_revenue", "first"),
            prediction_raw=("prediction_raw", "mean"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    raw_log = np.log(np.clip(grouped["prediction_raw"].to_numpy(), 1e-6, None))
    target_log = np.log(np.clip(grouped["actual_revenue"].to_numpy(), 1e-6, None))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_log, target_log)
    calibrator = RevenueCalibrator(
        x_thresholds=np.asarray(isotonic.X_thresholds_, dtype=float),
        y_thresholds=np.asarray(isotonic.y_thresholds_, dtype=float),
    )

    calibrated_oof = oof_predictions.copy()
    calibrated_oof["prediction_calibrated"] = calibrator.predict(
        calibrated_oof["prediction_raw"].to_numpy()
    )

    grouped["prediction_calibrated"] = calibrator.predict(grouped["prediction_raw"])
    summary = {
        "raw_rmse": metrics(
            grouped["actual_revenue"].to_numpy(),
            grouped["prediction_raw"].to_numpy(),
        )["RMSE"],
        "calibrated_rmse": metrics(
            grouped["actual_revenue"].to_numpy(),
            grouped["prediction_calibrated"].to_numpy(),
        )["RMSE"],
    }
    return RevenueCalibrationResult(
        calibrator=calibrator,
        grouped_frame=grouped,
        calibrated_oof=calibrated_oof,
        summary=summary,
    )
