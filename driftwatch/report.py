"""Terminal report rendering and distribution plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from driftwatch.monitor import driftLevelOrder


def renderTerminalReport(driftReport: dict) -> str:
    """Render the analysis as a readable plain-text report."""
    lines = ["", "DATA DRIFT REPORT", "=" * 17, ""]

    rowCounts = driftReport["schema"]["rowCounts"]
    lines.append("Rows:")
    lines.append(f"  Reference: {rowCounts['reference']:,}")
    lines.append(f"  Current:   {rowCounts['current']:,}")
    lines.append("")

    schemaReport = driftReport["schema"]
    lines.append("Schema:")
    lines.append(f"  {len(schemaReport['matchingColumns'])} matching columns")
    for column in schemaReport["missingColumns"]:
        lines.append(f"  ! missing column: {column}")
    for column in schemaReport["newColumns"]:
        lines.append(f"  ! new column: {column}")
    for column, (referenceDtype, currentDtype) in schemaReport["dtypeMismatches"].items():
        lines.append(f"  ! type mismatch: {column} ({referenceDtype} -> {currentDtype})")
    lines.append("")

    lines.append("Missing data:")
    for column, missingReport in driftReport["missingness"].items():
        warningText = "  WARNING" if missingReport["warning"] else ""
        lines.append(
            f"  {column:<14} ref {missingReport['referenceMissingRate']:>5.1%}"
            f"  cur {missingReport['currentMissingRate']:>5.1%}"
            f"  change {missingReport['change']:+.1%}{warningText}"
        )
    lines.append("")

    lines.append("Drift by feature:")
    for column, columnReport in driftReport["columns"].items():
        if columnReport["driftLevel"] == "UNKNOWN":
            lines.append(f"  {column:<14} UNKNOWN   ({columnReport['reason']})")
        elif columnReport["kind"] == "numeric":
            lines.append(
                f"  {column:<14} {columnReport['driftLevel']:<9}"
                f" KS {columnReport['ksStatistic']:.3f}  p {columnReport['pValue']:.3g}"
                f"  shift {columnReport['normalisedShift']:.2f} SD"
                f"  (mean {columnReport['referenceMean']:,.1f} -> {columnReport['currentMean']:,.1f})"
            )
        else:
            lines.append(
                f"  {column:<14} {columnReport['driftLevel']:<9}"
                f" max category share change {columnReport['maxShareChange']:.1%}"
            )
    lines.append("")

    summary = driftReport["summary"]
    for level in driftLevelOrder:
        flaggedColumns = summary["levels"][level]
        if flaggedColumns and level != "LOW":
            lines.append(f"{level}: {', '.join(flaggedColumns)}")
    lines.append(
        f"Overall: {summary['driftDetected']} / {summary['featuresAnalysed']}"
        " features show drift (HIGH or MODERATE)"
    )
    return "\n".join(lines)


def plotDistributions(
    referenceFrame: pd.DataFrame,
    currentFrame: pd.DataFrame,
    driftReport: dict,
    outputDirectory: str | Path,
) -> list[Path]:
    """Save an overlay plot for every feature flagged HIGH or MODERATE."""
    outputDirectory = Path(outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    savedPaths = []

    flaggedColumns = (
        driftReport["summary"]["levels"]["HIGH"] + driftReport["summary"]["levels"]["MODERATE"]
    )
    for column in flaggedColumns:
        columnReport = driftReport["columns"][column]
        figure, axes = plt.subplots(figsize=(7, 4))

        if columnReport["kind"] == "numeric":
            referenceValues = referenceFrame[column].dropna()
            currentValues = currentFrame[column].dropna()
            binEdges = np.histogram_bin_edges(
                np.concatenate([referenceValues, currentValues]), bins=40
            )
            axes.hist(referenceValues, bins=binEdges, alpha=0.6, density=True, label="reference")
            axes.hist(currentValues, bins=binEdges, alpha=0.6, density=True, label="current")
            axes.set_ylabel("Density")
        else:
            categories = list(columnReport["categoryShares"])
            referenceShares = [columnReport["categoryShares"][c]["reference"] for c in categories]
            currentShares = [columnReport["categoryShares"][c]["current"] for c in categories]
            positions = np.arange(len(categories))
            axes.bar(positions - 0.2, referenceShares, 0.4, label="reference")
            axes.bar(positions + 0.2, currentShares, 0.4, label="current")
            axes.set_xticks(positions)
            axes.set_xticklabels(categories)
            axes.set_ylabel("Share")

        axes.set_title(f"{column} ({columnReport['driftLevel']} drift)")
        axes.legend()
        figure.tight_layout()
        outputPath = outputDirectory / f"{column}.png"
        figure.savefig(outputPath, dpi=150)
        plt.close(figure)
        savedPaths.append(outputPath)

    return savedPaths
