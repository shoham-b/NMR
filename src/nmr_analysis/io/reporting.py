import json
from pathlib import Path
from typing import List
import pandas as pd
from nmr_analysis.core.types import AnalysisResult
from nmr_analysis.visualization.interactive import AnalysisContext


def save_report(result: AnalysisResult, output_dir: Path):
    """
    Save analysis results to the output directory.
    - results.json: Parameters and stats.
    - fit.png: Plot of the fit.
    - residuals.csv: Residuals data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    report_data = {
        "experiment_type": result.experiment_type.value,
        "dataset_name": result.dataset_name,
        "params": result.params,
        "r_squared": result.r_squared,
        "metadata": result.metadata,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # Save Residuals CSV
    df = pd.DataFrame({"residuals": result.residuals, "fit": result.fit_curve})
    df.to_csv(output_dir / "fit_data.csv", index=False)

    # Save Plot (re-generate it simply here or pass the figure?
    # For now, we assume this function is called alongside plotting or we rely on the passed data being enough to reconstruct if we had x/y.
    # But result object doesn't have x/y stored directly, only residuals/fit.
    # We should probably update AnalysisResult to store the input x/y if we want to replot here,
    # or rely on the caller to save the plot.

    # Let's assume the caller handles the plot saving for now or we add the plot logic here if we have x/y.
    pass


def save_summary_csv(
    contexts: List[AnalysisContext],
    output_dir: Path,
    filename: str = "summary.csv",
):
    """
    Save a summary CSV of all analysis results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ctx in contexts:
        res = ctx.result
        # Base row data
        row = {
            "Experiment": res.experiment_type.value,
            "Dataset": res.dataset_name,
            "R-Squared": res.r_squared,
        }

        # Add sample name if available
        if hasattr(ctx, "sample_name") and ctx.sample_name:
            row["Sample"] = ctx.sample_name

        # Flatten params
        for k, v in res.params.items():
            row[k] = v

        # Flatten param errors
        for k, v in res.param_errors.items():
            row[f"{k}_error"] = v

        # Add key metadata
        # e.g. T2_fixed if present
        if "fixed_t2" in res.metadata:
            row["fixed_t2"] = res.metadata["fixed_t2"]

        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Reorder columns to put identifiers first
    cols = list(df.columns)
    first_cols = ["Sample", "Experiment", "Dataset"]
    ordered_cols = [c for c in first_cols if c in cols] + [
        c for c in cols if c not in first_cols
    ]
    df = df[ordered_cols]

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
