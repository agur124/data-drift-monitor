"""Tests for schema comparison and monitor-level behaviour."""

import pandas as pd
import pytest

from driftwatch.monitor import analyseDrift
from driftwatch.schema import compareSchema


def test_identicalSchemasMatchCleanly():
    referenceFrame = pd.DataFrame({"age": [30, 40], "region": ["UK", "Europe"]})
    currentFrame = pd.DataFrame({"age": [35, 45], "region": ["UK", "Other"]})
    schemaReport = compareSchema(referenceFrame, currentFrame)
    assert schemaReport["matchingColumns"] == ["age", "region"]
    assert schemaReport["missingColumns"] == []
    assert schemaReport["newColumns"] == []
    assert schemaReport["dtypeMismatches"] == {}
    assert schemaReport["rowCounts"] == {"reference": 2, "current": 2}


def test_missingAndNewColumnsAreDetected():
    referenceFrame = pd.DataFrame({"age": [1], "postcode": ["OX1"]})
    currentFrame = pd.DataFrame({"age": [2], "device_version": ["v2"]})
    schemaReport = compareSchema(referenceFrame, currentFrame)
    assert schemaReport["missingColumns"] == ["postcode"]
    assert schemaReport["newColumns"] == ["device_version"]


def test_dtypeMismatchIsDetected():
    referenceFrame = pd.DataFrame({"age": [30, 40]})
    currentFrame = pd.DataFrame({"age": ["thirty", "forty"]})
    schemaReport = compareSchema(referenceFrame, currentFrame)
    assert "age" in schemaReport["dtypeMismatches"]


def test_intAndFloatColumnsAreCompatible():
    referenceFrame = pd.DataFrame({"age": [30, 40]})
    currentFrame = pd.DataFrame({"age": [30.5, 40.5]})
    assert compareSchema(referenceFrame, currentFrame)["dtypeMismatches"] == {}


def test_emptyDataRaisesValueError():
    populatedFrame = pd.DataFrame({"age": [1.0]})
    with pytest.raises(ValueError):
        analyseDrift(pd.DataFrame(), populatedFrame)
    with pytest.raises(ValueError):
        analyseDrift(populatedFrame, pd.DataFrame())


def test_mismatchedColumnIsExcludedFromDriftAnalysis():
    referenceFrame = pd.DataFrame({"age": [30.0, 40.0], "score": [1, 2]})
    currentFrame = pd.DataFrame({"age": [30.0, 41.0], "score": ["a", "b"]})
    driftReport = analyseDrift(referenceFrame, currentFrame)
    assert "score" not in driftReport["columns"]
    assert "age" in driftReport["columns"]
    assert "score" in driftReport["schema"]["dtypeMismatches"]
