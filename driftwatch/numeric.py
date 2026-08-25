"""Distribution drift measures for numeric columns.

Two complementary measures are used:

- The two-sample Kolmogorov-Smirnov statistic: the largest vertical gap
  between the two empirical cumulative distributions. Larger means the
  distribution shapes differ more. Its p-value is reported for information,
  but with thousands of rows even trivial differences become "significant",
  so classification is based on effect magnitude, never on the p-value.
- The Wasserstein distance: informally, how far probability mass must be
  moved to turn one distribution into the other, in the column's own units.
  Dividing it by the reference standard deviation gives a scale-free shift.
"""

import numpy as np
import pandas as pd
from scipy import stats

defaultNumericThresholds = {
    "moderateKs": 0.10,
    "highKs": 0.25,
    "moderateShift": 0.20,
    "highShift": 0.50,
}


def classifyNumericDrift(ksStatistic: float, normalisedShift: float, thresholds: dict) -> str:
    if ksStatistic >= thresholds["highKs"] or normalisedShift >= thresholds["highShift"]:
        return "HIGH"
    if ksStatistic >= thresholds["moderateKs"] or normalisedShift >= thresholds["moderateShift"]:
        return "MODERATE"
    return "LOW"


def compareNumericColumn(
    referenceValues: pd.Series, currentValues: pd.Series, thresholds: dict | None = None
) -> dict:
    """Compare one numeric column between the reference and current datasets."""
    thresholds = thresholds or defaultNumericThresholds
    referenceClean = np.asarray(referenceValues.dropna(), dtype=float)
    currentClean = np.asarray(currentValues.dropna(), dtype=float)

    if len(referenceClean) < 2 or len(currentClean) < 2:
        return {"driftLevel": "UNKNOWN", "reason": "fewer than two valid values"}

    ksStatistic, pValue = stats.ks_2samp(referenceClean, currentClean)
    wassersteinDistance = stats.wasserstein_distance(referenceClean, currentClean)

    referenceStd = referenceClean.std(ddof=1)
    if referenceStd > 0:
        normalisedShift = wassersteinDistance / referenceStd
    else:
        # A constant reference column: any movement at all is a definite shift.
        normalisedShift = 0.0 if wassersteinDistance == 0 else float("inf")

    return {
        "referenceMean": float(referenceClean.mean()),
        "currentMean": float(currentClean.mean()),
        "referenceStd": float(referenceStd),
        "currentStd": float(currentClean.std(ddof=1)),
        "ksStatistic": float(ksStatistic),
        "pValue": float(pValue),
        "wassersteinDistance": float(wassersteinDistance),
        "normalisedShift": float(normalisedShift),
        "driftLevel": classifyNumericDrift(float(ksStatistic), float(normalisedShift), thresholds),
    }
