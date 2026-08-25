"""Tests for categorical drift measures."""

import numpy as np
import pandas as pd
import pytest

from driftwatch.categorical import compareCategoricalColumn


def test_identicalSharesAreLow():
    referenceValues = pd.Series(["UK"] * 70 + ["Europe"] * 30)
    currentValues = pd.Series(["UK"] * 140 + ["Europe"] * 60)
    categoricalReport = compareCategoricalColumn(referenceValues, currentValues)
    assert categoricalReport["driftLevel"] == "LOW"
    assert categoricalReport["maxShareChange"] == pytest.approx(0.0)


def test_strongShiftIsHigh():
    referenceValues = pd.Series(["UK"] * 72 + ["Europe"] * 18 + ["Other"] * 10)
    currentValues = pd.Series(["UK"] * 51 + ["Europe"] * 29 + ["Other"] * 20)
    categoricalReport = compareCategoricalColumn(referenceValues, currentValues)
    assert categoricalReport["driftLevel"] == "HIGH"
    assert categoricalReport["maxShareChange"] == pytest.approx(0.21, abs=0.01)


def test_moderateShiftIsModerate():
    referenceValues = pd.Series(["UK"] * 70 + ["Europe"] * 30)
    currentValues = pd.Series(["UK"] * 62 + ["Europe"] * 38)
    assert compareCategoricalColumn(referenceValues, currentValues)["driftLevel"] == "MODERATE"


def test_newCategoryCountsFully():
    referenceValues = pd.Series(["mobile"] * 90 + ["desktop"] * 10)
    currentValues = pd.Series(["mobile"] * 80 + ["desktop"] * 10 + ["tablet"] * 10)
    categoricalReport = compareCategoricalColumn(referenceValues, currentValues)
    assert categoricalReport["categoryShares"]["tablet"]["reference"] == 0.0
    assert categoricalReport["categoryShares"]["tablet"]["current"] == pytest.approx(0.10)


def test_allMissingIsUnknown():
    emptyValues = pd.Series([np.nan, np.nan])
    assert compareCategoricalColumn(emptyValues, emptyValues)["driftLevel"] == "UNKNOWN"
