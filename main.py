"""Command-line interface for the data drift monitor.

Examples:

    python main.py data/reference.csv data/current_drifted.csv
    python main.py data/reference.csv data/current_drifted.csv --plots results
"""

import argparse

import pandas as pd

from driftwatch.monitor import analyseDrift
from driftwatch.report import plotDistributions, renderTerminalReport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a current dataset against a reference dataset and report drift."
    )
    parser.add_argument("referencePath", help="CSV file containing the reference data")
    parser.add_argument("currentPath", help="CSV file containing the current data")
    parser.add_argument(
        "--plots", metavar="DIRECTORY", help="also save distribution plots for drifted features"
    )
    arguments = parser.parse_args()

    referenceFrame = pd.read_csv(arguments.referencePath)
    currentFrame = pd.read_csv(arguments.currentPath)

    driftReport = analyseDrift(referenceFrame, currentFrame)
    print(renderTerminalReport(driftReport))

    if arguments.plots:
        savedPaths = plotDistributions(referenceFrame, currentFrame, driftReport, arguments.plots)
        if savedPaths:
            print("\nPlots saved:")
            for savedPath in savedPaths:
                print(f"  {savedPath}")
        else:
            print("\nNo features were flagged, so no plots were saved.")


if __name__ == "__main__":
    main()
