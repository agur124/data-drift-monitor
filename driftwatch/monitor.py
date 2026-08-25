"""Orchestrates schema, missingness and per-column drift analysis."""

import pandas as pd

from driftwatch.categorical import compareCategoricalColumn
from driftwatch.numeric import compareNumericColumn
from driftwatch.schema import compareSchema

missingChangeWarningThreshold = 0.02
driftLevelOrder = ("HIGH", "MODERATE", "LOW", "UNKNOWN")


def compareMissingness(
    referenceFrame: pd.DataFrame, currentFrame: pd.DataFrame, columns: list[str]
) -> dict:
    """Compare the missing-value rate of every shared column."""
    missingnessReport = {}
    for column in columns:
        referenceMissingRate = float(referenceFrame[column].isna().mean())
        currentMissingRate = float(currentFrame[column].isna().mean())
        change = currentMissingRate - referenceMissingRate
        missingnessReport[column] = {
            "referenceMissingRate": referenceMissingRate,
            "currentMissingRate": currentMissingRate,
            "change": change,
            "warning": abs(change) >= missingChangeWarningThreshold,
        }
    return missingnessReport


def analyseDrift(
    referenceFrame: pd.DataFrame,
    currentFrame: pd.DataFrame,
    numericThresholds: dict | None = None,
    categoricalThresholds: dict | None = None,
) -> dict:
    """Run the full drift analysis and return a structured report."""
    if referenceFrame.empty or currentFrame.empty:
        raise ValueError("both datasets must contain at least one row")

    schemaReport = compareSchema(referenceFrame, currentFrame)

    # Columns with incompatible types cannot be compared meaningfully, so they
    # are reported as schema problems and excluded from the drift analysis.
    analysableColumns = [
        column
        for column in schemaReport["matchingColumns"]
        if column not in schemaReport["dtypeMismatches"]
    ]
    missingnessReport = compareMissingness(referenceFrame, currentFrame, analysableColumns)

    columnReports = {}
    for column in analysableColumns:
        if pd.api.types.is_numeric_dtype(referenceFrame[column]):
            columnReports[column] = compareNumericColumn(
                referenceFrame[column], currentFrame[column], numericThresholds
            )
            columnReports[column]["kind"] = "numeric"
        else:
            columnReports[column] = compareCategoricalColumn(
                referenceFrame[column], currentFrame[column], categoricalThresholds
            )
            columnReports[column]["kind"] = "categorical"

    levelGroups = {level: [] for level in driftLevelOrder}
    for column, columnReport in columnReports.items():
        levelGroups[columnReport["driftLevel"]].append(column)

    return {
        "schema": schemaReport,
        "missingness": missingnessReport,
        "columns": columnReports,
        "summary": {
            "featuresAnalysed": len(columnReports),
            "driftDetected": len(levelGroups["HIGH"]) + len(levelGroups["MODERATE"]),
            "levels": levelGroups,
        },
    }
