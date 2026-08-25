"""Generates the controlled example datasets.

Three CSV files are written to data/:

- reference.csv        the baseline the monitor compares against
- current_normal.csv   drawn from the same distributions (should stay quiet)
- current_drifted.csv  deliberately shifted distributions (should be flagged)

Because the drift is planted by this script, the ground truth is known
exactly, which is what makes the validation experiment in the README possible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

dataDirectory = Path(__file__).resolve().parent / "data"

referenceRowCount = 10000
normalRowCount = 9500
driftedRowCount = 8400

regionCategories = ["UK", "Europe", "Other"]
deviceCategories = ["mobile", "desktop", "tablet"]


def generateFrame(
    randomSeed: int,
    rowCount: int,
    ageMean: float = 35.0,
    ageStd: float = 8.0,
    incomeMean: float = 35000.0,
    incomeStd: float = 12000.0,
    regionWeights: tuple = (0.72, 0.18, 0.10),
    missingIncomeRate: float = 0.012,
) -> pd.DataFrame:
    """Draw one synthetic customer dataset from the given distributions."""
    randomGenerator = np.random.default_rng(randomSeed)

    age = np.clip(randomGenerator.normal(ageMean, ageStd, rowCount), 18, 90).round(1)
    income = np.clip(randomGenerator.normal(incomeMean, incomeStd, rowCount), 0, None).round(2)
    region = randomGenerator.choice(regionCategories, size=rowCount, p=regionWeights)
    deviceType = randomGenerator.choice(deviceCategories, size=rowCount, p=(0.55, 0.35, 0.10))
    accountAge = np.clip(randomGenerator.exponential(600, rowCount), 0, 3650).round(0)

    missingMask = randomGenerator.random(rowCount) < missingIncomeRate
    income[missingMask] = np.nan

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "region": region,
            "device_type": deviceType,
            "account_age": accountAge,
        }
    )


def main() -> None:
    dataDirectory.mkdir(exist_ok=True)

    referenceFrame = generateFrame(randomSeed=101, rowCount=referenceRowCount)

    # Same distributions, different sample: the monitor should stay quiet.
    normalFrame = generateFrame(randomSeed=202, rowCount=normalRowCount)

    # Planted drift: older customers, higher incomes, altered region mix and
    # a jump in missing income values.
    driftedFrame = generateFrame(
        randomSeed=303,
        rowCount=driftedRowCount,
        ageMean=48.0,
        ageStd=10.0,
        incomeMean=49000.0,
        regionWeights=(0.51, 0.29, 0.20),
        missingIncomeRate=0.087,
    )

    for fileName, frame in (
        ("reference.csv", referenceFrame),
        ("current_normal.csv", normalFrame),
        ("current_drifted.csv", driftedFrame),
    ):
        outputPath = dataDirectory / fileName
        frame.to_csv(outputPath, index=False)
        print(f"Wrote {len(frame):,} rows to {outputPath}")


if __name__ == "__main__":
    main()
