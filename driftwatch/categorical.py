"""Distribution drift measures for categorical columns.

The first-version measure is deliberately simple: compare the share of each
category in the reference and current data and take the largest absolute
change. A category that appears or disappears entirely counts with its full
share. Jensen-Shannon distance is a natural later refinement.
"""

import pandas as pd

defaultCategoricalThresholds = {
    "moderateShare": 0.05,
    "highShare": 0.15,
}


def classifyCategoricalDrift(maxShareChange: float, thresholds: dict) -> str:
    if maxShareChange >= thresholds["highShare"]:
        return "HIGH"
    if maxShareChange >= thresholds["moderateShare"]:
        return "MODERATE"
    return "LOW"


def compareCategoricalColumn(
    referenceValues: pd.Series, currentValues: pd.Series, thresholds: dict | None = None
) -> dict:
    """Compare one categorical column between the reference and current datasets."""
    thresholds = thresholds or defaultCategoricalThresholds
    referenceShares = referenceValues.value_counts(normalize=True, dropna=True)
    currentShares = currentValues.value_counts(normalize=True, dropna=True)

    if referenceShares.empty and currentShares.empty:
        return {"driftLevel": "UNKNOWN", "reason": "no valid values on either side"}

    allCategories = sorted(set(referenceShares.index) | set(currentShares.index), key=str)
    categoryShares = {
        str(category): {
            "reference": float(referenceShares.get(category, 0.0)),
            "current": float(currentShares.get(category, 0.0)),
        }
        for category in allCategories
    }
    maxShareChange = max(
        abs(shares["current"] - shares["reference"]) for shares in categoryShares.values()
    )

    return {
        "categoryShares": categoryShares,
        "maxShareChange": float(maxShareChange),
        "driftLevel": classifyCategoricalDrift(float(maxShareChange), thresholds),
    }
