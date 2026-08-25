"""Schema comparison between the reference and current datasets."""

import pandas as pd

numericKinds = frozenset("iuf")


def dtypesAreCompatible(referenceDtype, currentDtype) -> bool:
    """Treat all numeric widths as compatible; otherwise require an exact match.

    An int column arriving as float (a common CSV artefact, especially once
    missing values appear) should not be reported as a schema break.
    """
    if referenceDtype.kind in numericKinds and currentDtype.kind in numericKinds:
        return True
    return referenceDtype == currentDtype


def compareSchema(referenceFrame: pd.DataFrame, currentFrame: pd.DataFrame) -> dict:
    """Report matching, missing and new columns, dtype mismatches and row counts."""
    matchingColumns = [column for column in referenceFrame.columns if column in currentFrame.columns]
    missingColumns = [column for column in referenceFrame.columns if column not in currentFrame.columns]
    newColumns = [column for column in currentFrame.columns if column not in referenceFrame.columns]

    dtypeMismatches = {
        column: (str(referenceFrame[column].dtype), str(currentFrame[column].dtype))
        for column in matchingColumns
        if not dtypesAreCompatible(referenceFrame[column].dtype, currentFrame[column].dtype)
    }

    return {
        "matchingColumns": matchingColumns,
        "missingColumns": missingColumns,
        "newColumns": newColumns,
        "dtypeMismatches": dtypeMismatches,
        "rowCounts": {"reference": len(referenceFrame), "current": len(currentFrame)},
    }
