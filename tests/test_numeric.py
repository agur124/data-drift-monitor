"""Tests for numeric drift measures."""

import numpy as np
import pandas as pd
import pytest

from driftwatch.numeric import compareNumericColumn

randomGenerator = np.random.default_rng(42)


def test_identicalDistributionsAreLow():
    referenceValues = pd.Series(randomGenerator.normal(0, 1, 3000))
    currentValues = pd.Series(randomGenerator.normal(0, 1, 3000))
    numericReport = compareNumericColumn(referenceValues, currentValues)
    assert numericReport["driftLevel"] == "LOW"
    assert numericReport["ksStatistic"] < 0.05


def test_strongShiftIsHigh():
    referenceValues = pd.Series(randomGenerator.normal(0, 1, 3000))
    currentValues = pd.Series(randomGenerator.normal(3, 1, 3000))
    numericReport = compareNumericColumn(referenceValues, currentValues)
    assert numericReport["driftLevel"] == "HIGH"
    assert numericReport["wassersteinDistance"] == pytest.approx(3.0, abs=0.2)


def test_nanValuesAreIgnored():
    referenceValues = pd.Series([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    currentValues = pd.Series([1.5, 2.5, np.nan, 3.5, 4.5])
    numericReport = compareNumericColumn(referenceValues, currentValues)
    assert numericReport["driftLevel"] in {"LOW", "MODERATE", "HIGH"}


def test_identicalConstantColumnsAreLow():
    constantValues = pd.Series([5.0] * 100)
    numericReport = compareNumericColumn(constantValues, constantValues)
    assert numericReport["driftLevel"] == "LOW"
    assert numericReport["wassersteinDistance"] == 0.0


def test_differingConstantColumnsAreHigh():
    numericReport = compareNumericColumn(pd.Series([5.0] * 100), pd.Series([9.0] * 100))
    assert numericReport["driftLevel"] == "HIGH"


def test_allNanColumnIsUnknown():
    numericReport = compareNumericColumn(pd.Series([np.nan, np.nan]), pd.Series([1.0, 2.0]))
    assert numericReport["driftLevel"] == "UNKNOWN"
