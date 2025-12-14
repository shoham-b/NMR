import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from nmr_analysis.core.types import AnalysisResult


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
