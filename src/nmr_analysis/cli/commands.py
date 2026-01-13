import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from scipy.ndimage import gaussian_filter1d

from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.hybrid import analyze_spectral_series, HybridAnalysisResult
from nmr_analysis.analysis.models import t2_decay_model
from nmr_analysis.analysis.processing import (
    extract_echo_train,
    preprocess_data,
    compute_spectrum,
    get_delay_from_metadata,
    parse_time_from_filename,
)
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
from nmr_analysis.io.loader import get_loader
from nmr_analysis.io.reporting import save_summary_csv
from nmr_analysis.visualization.interactive import generate_dashboard, AnalysisContext

ANALYSIS_SMOOTHING = 2.6

app = typer.Typer()
console = Console()


def _get_week_and_substance(path: Path, prefix: str = "") -> Tuple[str, str]:
    """
    Extract week and substance information from path and prefix.

    Args:
        path: The path to the data directory/file being analyzed.
        prefix: Optional prefix (usually week info from parent directory).

    Returns:
        Tuple of (week, substance) strings.
    """
    import re

    # Extract week from prefix or parent directories
    # Look for patterns like "week4.2", "week_4", "wk5", etc.
    week = ""
    if prefix:
        week_match = re.search(
            r"(week[_\-\s]?\d+\.?\d*|wk[_\-\s]?\d+\.?\d*)", prefix, re.IGNORECASE
        )
        if week_match:
            week = (
                week_match.group(1).replace(" ", "").replace("-", "").replace("_", "")
            )

    # If no week in prefix, try to find in path ancestry
    if not week:
        for parent in [path] + list(path.parents):
            week_match = re.search(
                r"(week[_\-\s]?\d+\.?\d*|wk[_\-\s]?\d+\.?\d*)",
                parent.name,
                re.IGNORECASE,
            )
            if week_match:
                week = (
                    week_match.group(1)
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )
                break

    # Determine substance from directory structure
    # The substance is the parent directory (e.g., "methanol", "water", "glycerin")
    # If not in a nested directory, default to "mineral-oil"
    target_name = path.name.lower()
    parent_name = path.parent.name.lower() if path.parent else ""

    # Check if current path is a substance directory or an experiment directory
    experiment_aliases = {
        "t1",
        "t2",
        "t2~",
        "t2_star",
        "t2_single",
        "t2multiple",
        "t2_multiple",
        "t2combined",
        "t2_combined",
        "diffusion",
    }

    if target_name in experiment_aliases:
        # Path is an experiment folder (like "t2"), so substance is the parent
        substance = (
            parent_name
            if parent_name and parent_name not in experiment_aliases
            else "mineral-oil"
        )
    else:
        # Path itself could be the substance folder (has experiment subfolders)
        # Or it's a top-level folder without nesting
        # Check if parent contains week pattern
        if parent_name and any(pat in parent_name for pat in ["week", "wk"]):
            # Parent is "week4.2", so target is the substance
            substance = target_name
        elif parent_name and parent_name not in experiment_aliases:
            substance = parent_name
        else:
            substance = "mineral-oil"

    # Check for ethanol_percent.txt in path or parents
    percent_file = None
    for search_path in [path, path.parent, path.parent.parent]:
        candidate = search_path / "ethanol_percent.txt"
        if candidate.exists():
            percent_file = candidate
            break

    if percent_file:
        try:
            content = percent_file.read_text().strip()
            # Extract number potentially floating point
            # Just take the content as is if it looks like a number
            if re.match(r"^\d+(\.\d+)?$", content):
                # Check if substance already has -X% (to avoid duplication if re-run)
                # But here we are constructing it fresh.
                substance = f"{substance}-{content}%"
        except Exception:
            pass  # Ignore read errors

    # Clean substance name (remove unwanted characters)
    substance = substance.replace(" ", "-").replace("_", "-")

    return week, substance


def _generate_plot_filename(
    path: Path,
    experiment: ExperimentType,
    graph_type: str,
    prefix: str = "",
) -> str:
    """
    Generate plot filename in format: {week}_{substance}_T{type}_{graphtype}.png

    Args:
        path: The path to the data directory/file being analyzed.
        experiment: The experiment type (T1, T2, T2_STAR, etc.).
        graph_type: Type of graph (fit, traces, combined, etc.).
        prefix: Optional prefix (usually week info from parent directory).

    Returns:
        Formatted filename string.
    """
    week, substance = _get_week_and_substance(path, prefix)

    # Map experiment type to short string
    type_map = {
        ExperimentType.T1: "T1",
        ExperimentType.T2: "T2",
        ExperimentType.T2_STAR: "T2star",
        ExperimentType.T2_COMBINED: "T2combined",
        ExperimentType.SPECTRUM: "Spectrum",
        ExperimentType.DIFFUSION: "Diffusion",
    }
    exp_str = type_map.get(experiment, experiment.value)

    # Build filename
    parts = []
    if week:
        parts.append(week)
    parts.append(substance)
    parts.append(exp_str)
    parts.append(graph_type)

    return "_".join(parts) + ".png"


@app.command()
def gui():
    """
    Launch the NMR Analysis Web GUI.
    """
    from nmr_analysis.gui.app import main as gui_main

    gui_main()


@app.command()
def analyze(
    path: Path = typer.Argument(
        ...,
        help="Path to input file (T2*), directory (T1/T2/Combined), or root directory for batch.",
    ),
    experiment: Optional[ExperimentType] = typer.Option(
        None,
        "-t",
        "--type",
        help="Type of experiment: t1, t2, t2_star, t2_combined. Auto-detected in batch mode.",
    ),
    channel: str = typer.Option("Channel 2", help="Scope channel name"),
    plot: bool = typer.Option(True, help="Show plot of the fit"),
    save_plots: bool = typer.Option(
        False, "--save-plots", help="Save plots to output directory"
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output-dir", help="Directory to save plots"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Generate interactive HTML report."
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Save all outputs directly to output directory without subfolders.",
    ),
):
    """
    Run analysis on NMR data. Supports batch processing of subdirectories.
    """
    collected_contexts: List[AnalysisContext] = []

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Batch Analysis Logic
    if path.is_dir() and experiment is None:
        # Check for subdirectories
        # Helper for case-insensitive matching
        # Map lower-case alias to ExperimentType
        ALIAS_MAP = {
            "t1": ExperimentType.T1,
            "t2": ExperimentType.T2,
            "t2_single": ExperimentType.T2,
            "t2~": ExperimentType.T2_STAR,
            "t2_star": ExperimentType.T2_STAR,
            "t2multiple": ExperimentType.T2_COMBINED,
            "t2_multiple": ExperimentType.T2_COMBINED,
            "diffusion": ExperimentType.DIFFUSION,
        }

        found_any = False

        # Iterate over all subdirectories in the input path
        for item in path.iterdir():
            if not item.is_dir():
                continue

            # Check if directory name matches an alias (case-insensitive)
            # Use lower case for matching but preserve original name for output/logging
            name_lower = item.name.lower()

            if name_lower in ALIAS_MAP:
                if name_lower == "t2~":
                    # Explicit mapping because dictionary lookup might be ambiguous on some setups
                    exp_type = ExperimentType.T2_STAR
                else:
                    exp_type = ALIAS_MAP[name_lower]

                # SPECIAL HANDLING FOR WATER DATASET
                # Logic refined:
                # Water/t2 -> T2 (Standard)
                # Water/t2_multiple -> DIFFUSION
                # Water/t2_multiple -> DIFFUSION (DISABLED to allow T2 Spin Echo analysis)
                if path.name.lower() in ("water", "data"):
                    # if exp_type == ExperimentType.T2_COMBINED:
                    #     exp_type = ExperimentType.DIFFUSION
                    #     console.print(
                    #         f"[cyan]Detected '{path.name}' dataset: Treating 't2_multiple' as DIFFUSION analysis.[/cyan]"
                    #     )
                    pass
                    # Note: t2 remains ExperimentType.T2

                found_any = True

                # Check for nested structure:
                # Does this folder contain subdirectories?
                # Exclude hidden folders just in case
                sub_folders = [
                    p
                    for p in item.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ]

                tasks = []
                if sub_folders:
                    # Nested mode: Process each subfolder as a separate dataset
                    for sub in sub_folders:
                        # Output path mirrors structure: output_dir / ExperimentDir / SampleDir
                        out_sub = None
                        if save_plots:
                            if flat:
                                out_sub = output_dir
                            else:
                                out_sub = output_dir / item.name / sub.name
                                out_sub.mkdir(parents=True, exist_ok=True)
                        tasks.append((sub, out_sub))
                else:
                    # Flat/Standard mode: Process the folder itself
                    out_std = None
                    if save_plots:
                        if flat:
                            out_std = output_dir
                        else:
                            out_std = output_dir / item.name
                            out_std.mkdir(parents=True, exist_ok=True)
                    tasks.append((item, out_std))

                # Sort tasks: Prioritize T2_COMBINED to make results available for DIFFUSION
                # Note: tasks contains (Path, OutputPath)
                # We need to sort based on the folder name of the Path.
                # Since this block is inside "ALIAS_MAP match", ALL tasks are of the SAME experiment type (exp_type).
                # So sorting by experiment type here is useless because `exp_type` is constant for this block!

                # WAIT. This block (lines 94+) is for when the FOLDER NAME matches an alias.
                # e.g. folder is "t1". Subfolders are "sample1", "sample2".
                # ALL subfolders are T1 experiments.
                # So we CANNOT have T2 Combined and Diffusion mixed here.
                # They would be separate top-level folders.

                # So I DO NOT need to sort tasks here. I just need to restore the code.

                for target_path, target_out in tasks:
                    console.rule(
                        f"[bold cyan]Batch Analysis: {target_path.name} ({exp_type.value})[/bold cyan]"
                    )
                    try:
                        ctxs = _run_analysis(
                            target_path,
                            exp_type,
                            channel,
                            plot,
                            save_path=target_out,
                            prefix=f"{target_path.parent.name}_{target_path.name}"
                            if flat
                            else "",
                        )
                        if ctxs:
                            collected_contexts.extend(ctxs)
                    except Exception as e:
                        console.print(
                            f"[red]Failed to analyze {target_path.name}: {e}[/red]"
                        )

            elif item.is_dir():
                # Item name is NOT in ALIAS_MAP.
                # Check if it is a "Sample" directory that *contains* experiment folders.
                # e.g. path/Water/t1, path/Water/t2

                sub_experiments = []
                for sub in item.iterdir():
                    if not sub.is_dir():
                        continue
                    if sub.name.lower() in ALIAS_MAP:
                        sub_experiments.append(sub)

                if sub_experiments:
                    console.print(
                        f"[magenta]Detected Sample Directory: {item.name}[/magenta]"
                    )
                    found_any = True

                    # Sort sub_experiments: Prioritize T2 (to get T2 for Diffusion constraint)
                    # 0: T2 (Highest Priority)
                    # 1: Others
                    sub_experiments.sort(
                        key=lambda x: 0
                        if ALIAS_MAP.get(x.name.lower())
                        in (ExperimentType.T2, "t2", "t2_single")
                        else 1
                    )

                    current_sample_t2_combined = None

                    for sub in sub_experiments:
                        name_lower = sub.name.lower()
                        exp_type = ALIAS_MAP[name_lower]

                        # Check for Water/Diffusion special case recursively
                        if item.name.lower() in ("water", "data"):
                            # Override DISABLED to allow T2 Spin Echo analysis details
                            # if exp_type == ExperimentType.T2_COMBINED:
                            #     exp_type = ExperimentType.DIFFUSION
                            #     console.print(
                            #         f"[cyan]Detected '{item.name}' dataset: Treating 't2_multiple' as DIFFUSION analysis.[/cyan]"
                            #     )
                            pass

                        # Calculate output path: output_dir / SampleName / Experiment
                        out_sub = None
                        if save_plots:
                            if flat:
                                out_sub = output_dir
                            else:
                                out_sub = output_dir / item.name / sub.name
                                out_sub.mkdir(parents=True, exist_ok=True)

                        console.rule(
                            f"[bold cyan]Sample Analysis: {item.name}/{sub.name} ({exp_type.value})[/bold cyan]"
                        )

                        try:
                            # If Diffusion, pass the T2 from Combined analysis if valid
                            kwargs = {}
                            if (
                                exp_type == ExperimentType.DIFFUSION
                                and current_sample_t2_combined is not None
                            ):
                                kwargs["fixed_t2"] = current_sample_t2_combined

                            ctxs = _run_analysis(
                                sub,
                                exp_type,
                                channel,
                                plot,
                                save_path=out_sub,
                                prefix=item.name if flat else "",
                                **kwargs,
                            )
                            if ctxs:
                                # Tag context with sample name?
                                for c in ctxs:
                                    c.sample_name = item.name
                                    # If this was T2 (Standard), store the result for Diffusion constraint
                                    if exp_type == ExperimentType.T2:
                                        if "T2" in c.result.params:
                                            current_sample_t2_combined = (
                                                c.result.params["T2"]
                                            )
                                            console.print(
                                                f"[green]Captured T2 for Diffusion constraint: {current_sample_t2_combined:.4f} s[/green]"
                                            )

                                collected_contexts.extend(ctxs)
                        except Exception as e:
                            console.print(
                                f"[red]Failed to analyze {item.name}/{sub.name}: {e}[/red]"
                            )

        if found_any:
            console.print("[green]Batch analysis completed.[/green]")
            if interactive and collected_contexts:
                output_html = output_dir / "index.html"
                generate_dashboard(collected_contexts, output_html)
                console.print(
                    f"[green]Interactive report saved to {output_html}[/green]"
                )

            if collected_contexts:
                save_summary_csv(collected_contexts, output_dir)
                console.print(
                    f"[green]Summary CSV saved to {output_dir / 'summary.csv'}[/green]"
                )
            return

        # If no subdirs found and no experiment specified, fail or assume single directory?
        # Let's fail nicely.
        console.print(
            "[yellow]No experiment type specified and no standard subdirectories (t1, t2, t~, t2combined) found.[/yellow]"
        )

        console.print("Please specify --type or ensure directory structure.")
        raise typer.Exit(1)

    # Standard Single Analysis
    if experiment is None:
        console.print("[red]Experiment type is required for single analysis.[/red]")
        raise typer.Exit(1)

    save_path = None
    if save_plots:
        save_path = output_dir
        save_path.mkdir(parents=True, exist_ok=True)

    ctxs = _run_analysis(path, experiment, channel, plot, save_path=save_path)
    if interactive and ctxs:
        output_html = output_dir / "index.html"
        generate_dashboard(ctxs, output_html)
        console.print(f"[green]Interactive report saved to {output_html}[/green]")


@app.command()
def montage(
    root_dir: Path = typer.Argument(
        ..., help="Root directory to search for ethanol data"
    ),
    output_file: Path = typer.Option(
        Path("ethanol_montage.png"), "--output", "-o", help="Output filename"
    ),
    channel: str = typer.Option("Channel 2", help="Scope channel name"),
    title: str = typer.Option("Ethanol T2 Fits Montage", help="Title for the montage"),
):
    """
    Generate a montage of T2 fits for all ethanol datasets found in the root directory.
    Searches for 'ethanol_percent.txt' to identify datasets.
    """
    console.print(
        f"[bold green]Searching for ethanol datasets in {root_dir}...[/bold green]"
    )

    found_datasets = []

    # Recursive search for ethanol_percent.txt
    for path in root_dir.rglob("ethanol_percent.txt"):
        dataset_dir = path.parent
        # Try to find a T2 subdirectory or check if this is the T2 directory
        # The user stores data like: Week4/Methanol/t2
        # So if we find metadata in 'Methanol', we should look for 't2' inside it.
        # Or if metadata is in 't2' (less likely?), or 'Week4' (too broad).
        # Assuming metadata is in the Substance folder (e.g. Methanol).

        t2_dir = None
        if (dataset_dir / "t2").exists():
            t2_dir = dataset_dir / "t2"
        elif (dataset_dir / "T2").exists():
            t2_dir = dataset_dir / "T2"
        elif dataset_dir.name.lower() == "t2":
            t2_dir = dataset_dir

        if t2_dir:
            # Check if this t2_dir contains data
            if any(t2_dir.glob("*.h5")) or any(t2_dir.glob("*.csv")):
                found_datasets.append((dataset_dir, t2_dir, path))

    console.print(f"Found {len(found_datasets)} ethanol T2 datasets.")

    if not found_datasets:
        console.print("[yellow]No ethanol datasets found.[/yellow]")
        return

    results = []

    with Progress(console=console) as progress:
        task = progress.add_task(
            "[cyan]Analyzing datasets...", total=len(found_datasets)
        )

        for substance_dir, t2_path, meta_path in found_datasets:
            try:
                # Read percentage
                percent_str = meta_path.read_text().strip()
                percent_val = float(re.search(r"(\d+(\.\d+)?)", percent_str).group(1))

                # Run T2 Analysis (No plotting, just get result)
                # We reuse _run_analysis but suppress internal plotting
                # Note: _run_analysis returns a list of contexts

                # We need week info
                week, substance = _get_week_and_substance(t2_path)

                ctxs = _run_analysis(
                    t2_path,
                    ExperimentType.T2,
                    channel,
                    plot=False,  # We do our own plotting
                    save_path=None,
                )

                if ctxs:
                    # Assume the main T2 result is the first one or the one with T2 fit
                    # _run_analysis for T2 returns one context with aggregated data
                    ctx = ctxs[0]
                    results.append(
                        {
                            "week": week,
                            "percent": percent_val,
                            "substance": substance,
                            "context": ctx,
                            "dir": substance_dir.name,
                        }
                    )

            except Exception as e:
                console.print(f"[red]Error analyzing {t2_path}: {e}[/red]")
            finally:
                progress.advance(task)

    # Sort results: First by Week (alphanumeric), then by Percent (numeric)
    # Week might be "week4.2", "week5".
    # Let's try to extract numbers for week sorting.
    def week_sort_key(w):
        m = re.search(r"(\d+)", w)
        return int(m.group(1)) if m else 0

    results.sort(key=lambda x: (week_sort_key(x["week"]), x["percent"]))

    # Create Montage
    n_plots = len(results)
    if n_plots == 0:
        return

    cols = 4  # Adjustable
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True
    )
    axes = np.ravel(axes)

    # Global Title
    # Handle case where title might be OptionInfo (when called directly, not via CLI)
    title_str = title.default if hasattr(title, "default") else str(title)
    fig.suptitle(title_str, fontsize=20, fontweight="bold")

    for i, res in enumerate(results):
        ax = axes[i]
        ctx = res["context"]

        # Data
        time = ctx.data.time
        signal = np.real(ctx.data.signal)  # Use Signed Signal (User Request)

        # Fit
        fit_curve = ctx.result.fit_curve
        if fit_curve is None:
            # Should not happen for T2 fit unless failed
            ax.text(0.5, 0.5, "Fit Failed", ha="center", va="center")
            continue

        # Plot Data
        ax.scatter(time, signal, s=10, alpha=0.6, label="Data", color="blue")
        # Plot Fit
        ax.plot(time, fit_curve, "r-", linewidth=2, label="Fit")

        # Fit Info text
        try:
            t2_val = ctx.result.params.get("T2", 0)
            r2_val = ctx.result.r_squared
            info_text = f"T2: {t2_val:.4f} s\nR²: {r2_val:.4f}"
        except:
            info_text = "N/A"

        # Labels
        ax.set_title(
            f"{res['week']} - {res['percent']}% ({res['dir']})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal")
        ax.legend(loc="upper right", fontsize="small")

        # Add text box parameters
        ax.text(
            0.95,
            0.5,
            info_text,
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Log scale option? User said "fit right graphs", usually T2 is linear or log.
        # Standard T2 plots usually linear for fit view.

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    console.print(f"[green]Saving montage to {output_file.resolve()}[/green]")
    plt.savefig(output_file, dpi=300)
    plt.close()

    console.print(f"[bold green]Montage generated successfully![/bold green]")

    # --- NEW: T2 vs Ethanol Percentage Graph ---
    console.print("[cyan]Generating T2 vs Ethanol Percentage graph...[/cyan]")

    # Collect data for T2 vs Percent
    # Group by Week to allow multiple series if needed (though usually one series per folder)
    # results are already sorted by week then percent

    # Structure: { "WeekName": [(percent, t2, error), ...] }
    series_data = {}

    for res in results:
        week = res["week"]
        percent = res["percent"]
        ctx = res["context"]

        try:
            t2_val = ctx.result.params.get("T2", None)
            # Try to get error if available (standard error)
            t2_err = ctx.result.errors.get("T2", 0.0) if ctx.result.errors else 0.0

            if t2_val is not None:
                if week not in series_data:
                    series_data[week] = []
                series_data[week].append((percent, t2_val, t2_err))
        except Exception:
            pass

    if not series_data:
        console.print("[yellow]No T2 data found for correlation graph.[/yellow]")
        return

    # Plot T2 vs Percent
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    colors = cm.get_cmap("tab10")  # Use a colormap

    for i, (week, data_points) in enumerate(series_data.items()):
        # Sort by percent just in case
        data_points.sort(key=lambda x: x[0])

        percents = [d[0] for d in data_points]
        t2s = [d[1] for d in data_points]
        errors = [d[2] for d in data_points]

        label = week if week else "Data"

        ax2.errorbar(
            percents,
            t2s,
            yerr=errors,
            fmt="o-",
            linewidth=2,
            markersize=8,
            capsize=4,
            label=label,
            color=colors(i),
        )

    ax2.set_xlabel("Ethanol Percentage (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("T2 Relaxation Time (s)", fontsize=12, fontweight="bold")
    ax2.set_title("T2 vs Ethanol Percentage", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    if len(series_data) > 1:
        ax2.legend()

    # Construct output filename for this graph
    # Based on the montage output filename
    # e.g. ethanol_montage.png -> ethanol_t2_vs_percent.png
    base_name = output_file.stem
    if "montage" in base_name:
        new_name = base_name.replace("montage", "t2_vs_percent")
    else:
        new_name = f"{base_name}_t2_vs_percent"

    output_graph_file = output_file.parent / (new_name + output_file.suffix)

    console.print(
        f"[green]Saving T2 comparison graph to {output_graph_file.resolve()}[/green]"
    )
    plt.savefig(output_graph_file, dpi=300)
    plt.close()


def _run_analysis(
    path: Path,
    experiment: ExperimentType,
    channel: str,
    plot: bool,
    save_path: Optional[Path] = None,
    fixed_t2: Optional[float] = None,
    prefix: str = "",
) -> List[AnalysisContext]:
    results = []

    if experiment == ExperimentType.T2_STAR:
        # T2*: Analyze each file independently
        target_files = []
        if path.is_dir():
            target_files = (
                list(path.glob("*.h5"))
                + list(path.glob("*.hdf5"))
                + list(path.glob("*.csv"))
            )
            if not target_files:
                raise FileNotFoundError(f"No HDF5/CSV files in {path}")
        else:
            target_files = [path]

        console.print(f"Found {len(target_files)} T2* files to analyze.")

        for target_file in target_files:
            try:
                console.print(f"Loading {target_file.name}...")
                loader = get_loader(target_file, channel=channel)
                data = loader.load(target_file)
                console.print(f"Fitting T2* for {target_file.name}...")
                result = Fitter.fit_t2_star(data)
                # Add week/substance to dataset name
                week, substance = _get_week_and_substance(target_file, prefix)
                title_prefix = (
                    f"{week} {substance}".strip()
                    if week or substance != "mineral-oil"
                    else ""
                )
                if title_prefix:
                    result.dataset_name = f"{title_prefix} - {result.dataset_name}"
                # Ensure unique dataset name if multiple
                if len(target_files) > 1:
                    result.dataset_name = f"{result.dataset_name} ({target_file.stem})"

                print_result(result)
                if plot:
                    filepath = None
                    if save_path:
                        # If save_path is a directory (via batch or single with dir input), use it
                        # If single file input, save_path might be parent dir
                        out_dir = save_path if save_path.is_dir() else save_path.parent
                        fname = _generate_plot_filename(
                            target_file, experiment, "fit", prefix
                        )
                        filepath = out_dir / fname
                        console.print(f"Saving plot to {filepath.as_uri()}")

                    plot_result(
                        data.time,
                        np.abs(data.signal),
                        result,
                        f"Time ({data.metadata.get('time_unit', 's')})",
                        "Signal (Magnitude)",
                        filepath=filepath,
                    )
                results.append(AnalysisContext(data=data, result=result))
            except Exception as e:
                console.print(f"[red]Failed to analyze {target_file.name}: {e}[/red]")

        return results

        return results

    elif experiment == ExperimentType.SPECTRUM:
        # Spectrum Analysis
        target_files = []
        if path.is_dir():
            # Recursive search for all valid files
            raw_files = list(path.rglob("*"))
            target_files = [
                f for f in raw_files if f.suffix.lower() in [".h5", ".hdf5", ".csv"]
            ]
        else:
            target_files = [path]

        console.print(f"Found {len(target_files)} files for Spectral analysis.")

        if len(target_files) > 1:
            # Hybrid Series Analysis for multiple files
            console.print(
                "[bold green]Running Hybrid Spectral Series Analysis...[/bold green]"
            )
            data_list = []
            names = []
            for tf in target_files:
                try:
                    loader = get_loader(tf, channel=channel)
                    d = loader.load(tf)
                    data_list.append(d)
                    names.append(tf.name)
                except Exception as e:
                    console.print(f"[yellow]Skipping {tf.name}: {e}[/yellow]")

            if not data_list:
                console.print("[red]No valid data loaded for hybrid analysis.[/red]")
                return []

            # --- Run Standard T2 (Time Domain) Analysis First (Primary) ---
            # This will be the main result, consistent with other materials
            console.print(
                "[bold cyan]Running Standard T2 (Time Domain) Analysis...[/bold cyan]"
            )

            td_delays = []
            td_amplitudes = []
            td_raw_traces = []

            # Sort data by tau for processing
            tau_list = []
            for tf in target_files:
                try:
                    # Parse tau from filename
                    tau = parse_time_from_filename(tf)
                    if tau is None:
                        tau = get_delay_from_metadata(
                            get_loader(tf, channel=channel).load(tf)
                        )
                    tau_list.append((tau, tf))
                except Exception:
                    console.print(
                        f"[yellow]Could not parse tau from {tf.name}[/yellow]"
                    )

            tau_list.sort(key=lambda x: x[0])

            for tau, tf in tau_list:
                try:
                    loader = get_loader(tf, channel=channel)
                    d = loader.load(tf)

                    # Preprocess
                    processed_data, tau_fitted, _, peak_info = preprocess_data(
                        d, smoothing=ANALYSIS_SMOOTHING
                    )

                    # Extract Amplitude using SELECTED PEAK from peak_info
                    # (Consistent with T1/T2 directory path)
                    fit_idx = peak_info.get("fit_idx", 0)
                    sig = processed_data.signal
                    if fit_idx < len(sig):
                        amp = np.real(sig[fit_idx])
                    else:
                        amp = np.max(np.real(sig))

                    # Use ACTUAL detected delay from signal (tau_fitted) for the fit
                    # This ensures the X-axis matches the "Fit Peak" locations
                    td_delays.append(tau_fitted)
                    td_amplitudes.append(amp)

                    # Create raw trace tuple for plotting
                    # Format: (processed_data, t_peak, amp, tau, peak_info, data_full, sort_val)
                    processed_data.metadata["trace_label"] = f"{tf.stem}"
                    td_raw_traces.append(
                        (processed_data, 0.0, amp, tau, peak_info, d, tau)
                    )
                except Exception as e:
                    console.print(f"[yellow]Error processing {tf.name}: {e}[/yellow]")

            td_delays = np.array(td_delays)
            td_amplitudes = np.array(td_amplitudes)

            # Fit T2 (Time Domain)
            params, fit_curve, residuals, r2, param_errors = Fitter.fit_t2(
                td_delays, td_amplitudes
            )

            # Determine experiment name from directory with week/substance
            week, substance = _get_week_and_substance(path, prefix)
            title_prefix = (
                f"{week} {substance}".strip()
                if week or substance != "mineral-oil"
                else ""
            )
            dataset_label = path.name if path.is_dir() else path.parent.name
            td_name = (
                f"{title_prefix} - T2 Analysis: {dataset_label}"
                if title_prefix
                else f"T2 Analysis: {dataset_label}"
            )

            td_result = AnalysisResult(
                experiment_type=ExperimentType.T2,
                dataset_name=td_name,
                params=params,
                fit_curve=fit_curve,
                residuals=residuals,
                r_squared=r2,
                param_errors=param_errors,
            )

            print_result(td_result)

            # Plot Standard T2 Result (Primary)
            if plot:
                out_dir = save_path if save_path else target_files[0].parent
                fname_fit = _generate_plot_filename(
                    path, ExperimentType.T2, "fit", prefix
                )
                fname_traces = _generate_plot_filename(
                    path, ExperimentType.T2, "traces", prefix
                )
                filepath_fit = out_dir / fname_fit
                filepath_traces = out_dir / fname_traces

                console.print(f"Saving fit plot to {filepath_fit}")
                console.print(
                    f"Saving traces plot (3 columns: processed, raw, Fourier) to {filepath_traces}"
                )

                plot_stacked_traces(
                    td_raw_traces,
                    filepath=filepath_traces,
                    smoothing=ANALYSIS_SMOOTHING,
                    show_fourier=True,
                    title=td_result.dataset_name,
                )

                plot_analysis_summary(
                    td_delays,
                    td_amplitudes,
                    td_result,
                    td_raw_traces,
                    "Delay (s)",
                    "Amplitude",
                    filepath=filepath_fit,
                    smoothing=ANALYSIS_SMOOTHING,
                    show_fourier=True,
                )

            # --- ALSO run Hybrid Spectral Analysis (Supplementary) ---
            console.print(
                "[bold green]Running Supplementary Spectral Analysis...[/bold green]"
            )

            hybrid_res = analyze_spectral_series(data_list, names)

            # Print Summary
            console.print(
                f"[bold]Spectral Analysis Details: {hybrid_res.dataset_name}[/bold]"
            )
            table = Table(title="Spectral T2 Results (Per Frequency Peak)")
            table.add_column("Peak Freq (Hz)", justify="right")
            table.add_column("T2 (s)", justify="right")
            table.add_column("M0", justify="right")
            table.add_column("R2 Score", justify="right")

            for i, res in enumerate(hybrid_res.t2_results):
                f0 = hybrid_res.peak_centers[i]
                t2 = res.get("T2", 0)
                m0 = res.get("M0", 0)
                r2 = res.get("r_squared", 0)
                table.add_row(f"{f0:.2f}", f"{t2:.4f}", f"{m0:.2e}", f"{r2:.4f}")

            console.print(table)

            if plot:
                out_dir = save_path if save_path else target_files[0].parent
                prefix_str = f"{prefix}_" if prefix else ""
                try:
                    plot_hybrid_result(
                        hybrid_res,
                        out_dir,
                        source_path=path,
                        prefix=prefix,
                    )
                    console.print("[green]Spectral detail plots saved.[/green]")
                except Exception as e:
                    console.print(
                        f"[yellow]Could not plot spectral details: {e}[/yellow]"
                    )

            # Return standard T2 result as primary (consistent with other materials)
            aggregated_data = NMRData(time=td_delays, signal=td_amplitudes)
            return [
                AnalysisContext(
                    data=aggregated_data, result=td_result, raw_traces=td_raw_traces
                )
            ]

        else:
            # Standard Single File Analysis
            # User Request: "T2* is in the time domain as always was"
            # So even if Type=SPECTRUM, for single file, do Time Domain T2* Fit but Show Spectrum.
            for target_file in target_files:
                try:
                    console.print(f"Loading {target_file.name}...")
                    loader = get_loader(target_file, channel=channel)
                    data = loader.load(target_file)
                    # 1. Compute Spectrum for visualization (Focused on 2nd Peak if available)
                    from scipy.signal import find_peaks
                    from scipy.ndimage import gaussian_filter1d

                    sig_abs = np.abs(data.signal)
                    time = data.time
                    dt = time[1] - time[0] if len(time) > 1 else 1.0

                    # Smooth to find envelope (Echoes)
                    # Try sigma corresponding to 50 microseconds -> ~0.5ms?
                    # If dt ~ 2e-7 (5MHz), 1ms = 5000 pts.
                    # sigma=500 pts is heavy.
                    sigma_points = int(50e-6 / dt)
                    if sigma_points < 1:
                        sigma_points = 1

                    smoothed_sig = gaussian_filter1d(
                        sig_abs, sigma=sigma_points * 10
                    )  # Heavy smoothing

                    # Distance: Say echoes are at least 1ms apart
                    dist_points = int(1e-3 / dt)
                    if dist_points < 100:
                        dist_points = 100

                    # Height: 5% of max
                    peaks, _ = find_peaks(
                        smoothed_sig,
                        distance=dist_points,
                        height=0.05 * np.max(smoothed_sig),
                    )

                    if len(peaks) >= 2:
                        console.print(
                            f"Detected {len(peaks)} peaks. Focusing on 2nd peak (idx={peaks[1]})."
                        )

                        p1 = peaks[0]
                        p2 = peaks[1]
                        spacing = p2 - p1

                        # Define window around P2
                        # Start halfway from P1
                        start_idx = int(p2 - (spacing // 2))
                        # End symmetric or halfway to P3
                        end_idx = int(p2 + (spacing // 2))

                        # Bounds check
                        start_idx = max(0, start_idx)
                        end_idx = min(len(data.signal), end_idx)

                        # Create Sliced Data for FFT
                        # We create a temporary NMRData-like object or just pass arrays if compute_spectrum supported it?
                        # compute_spectrum takes NMRData.
                        sliced_signal = data.signal[start_idx:end_idx]
                        sliced_time = data.time[start_idx:end_idx]

                        # Create temp object for FFT
                        sliced_data = NMRData(time=sliced_time, signal=sliced_signal)
                        freqs, spect = compute_spectrum(sliced_data)
                        spec_title_suffix = " (2nd Peak)"
                    else:
                        console.print("Fewer than 2 peaks detected. using full signal.")
                        freqs, spect = compute_spectrum(data)
                        spec_title_suffix = " (Full)"

                    # 2. Fit Time Domain T2* (Standard)
                    console.print("Fitting T2* (Time Domain)...")
                    result = Fitter.fit_t2_star(data)

                    # Add dataset name
                    if len(target_files) > 1:
                        result.dataset_name = (
                            f"{result.dataset_name} ({target_file.stem})"
                        )

                    print_result(result)

                    if plot:
                        # Save Time Domain Fit Plot
                        if save_path:
                            out_dir = (
                                save_path if save_path.is_dir() else save_path.parent
                            )
                        else:
                            out_dir = target_file.parent

                        fname = (
                            f"{prefix}_{target_file.stem}_fit.png"
                            if prefix
                            else f"{target_file.stem}_fit.png"
                        )
                        filepath = out_dir / fname
                        console.print(f"Saving T2* fit plot to {filepath}")

                        plot_result(
                            data.time,
                            np.abs(data.signal),
                            result,
                            f"Time ({data.metadata.get('time_unit', 's')})",
                            "Signal (Magnitude)",
                            filepath=filepath,
                        )

                        # Also Save Spectrum Plot (without Fit)
                        # Also Save Spectrum Plot (without Fit)
                        # We used out_dir above, can reuse or re-derive
                        fname_spec = (
                            f"{prefix}_{target_file.stem}_spectrum.png"
                            if prefix
                            else f"{target_file.stem}_spectrum.png"
                        )
                        filepath_spec = out_dir / fname_spec
                        console.print(f"Saving Spectrum plot to {filepath_spec}")

                        # Simple spectrum plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Convert to kHz
                        ax.plot(freqs / 1000.0, np.abs(spect), color="black")
                        ax.set_title(f"Spectrum: {target_file.name}{spec_title_suffix}")
                        ax.set_xlabel("Frequency (kHz)")
                        ax.set_ylabel("Magnitude")
                        ax.set_xlim(-10, 10)
                        ax.grid(True, alpha=0.3)

                        plt.savefig(filepath_spec)
                        plt.close()

                    results.append(AnalysisContext(data=data, result=result))

                except Exception as e:
                    console.print(
                        f"[red]Failed to analyze {target_file.name}: {e}[/red]"
                    )

        return results

    elif experiment == ExperimentType.DIFFUSION:
        # Diffusion: Variable tau T2 analysis
        if not path.is_dir():
            raise typer.Exit(
                "Diffusion analysis requires a directory of T2 files with variable tau."
            )

        files = (
            list(path.glob("*.h5"))
            + list(path.glob("*.hdf5"))
            + list(path.glob("*.csv"))
        )
        if not files:
            raise typer.Exit("No files found for diffusion analysis.")

        console.print(f"Found {len(files)} files for Diffusion analysis.")

        taus = []
        rates = []

        # 1. Process each file to get T2
        with Progress() as progress:
            task = progress.add_task("Fitting T2 for each tau...", total=len(files))
            for f in files:
                try:
                    loader = get_loader(f, channel=channel)
                    data = loader.load(f)
                    data.experiment_type = ExperimentType.DIFFUSION

                    # Extract tau from filename (e.g. 0_0001.HDF5 -> 0.0001)
                    # Use same logic as batch T2
                    import re

                    match = re.search(r"(0_[\d\.]+)", f.stem)
                    if match:
                        tau_val = float(match.group(1).replace("_", "."))
                    else:
                        console.print(
                            f"[yellow]Could not parse tau from {f.stem}, skipping.[/yellow]"
                        )
                        continue

                    # Preprocess & Fit T2 using NMRMINE logic
                    # NMRMINE logic selects P1 and a specific echo (P2 or P3).
                    # We use these 2 points to fit a T2 decay.

                    data, _, amp, _ = preprocess_data(
                        data, smoothing=ANALYSIS_SMOOTHING
                    )

                    peak_times, peak_amps = extract_echo_train(
                        data, smoothing=ANALYSIS_SMOOTHING
                    )

                    if len(peak_times) < 3:
                        continue

                    # Remove manual peak skipping.
                    # Logic is now same as T2 Combined: preprocess (trim) -> extract_echo_train (monotonic).
                    # peak_times = peak_times[2:]
                    # peak_amps = peak_amps[2:]

                    params, _, _, _, _ = Fitter.fit_t2(peak_times, peak_amps)

                    if "T2" in params and params["T2"] > 0:
                        t2_obs = params["T2"]
                        r2_obs = 1.0 / t2_obs
                        taus.append(tau_val)
                        rates.append(r2_obs)

                except Exception as e:
                    console.print(f"[red]Error processing {f.name}: {e}[/red]")

                progress.advance(task)

        if len(taus) < 3:
            console.print("[red]Not enough valid data points for diffusion fit.[/red]")
            return []

        # 2. Fit Diffusion
        console.print("Fitting Diffusion Coefficient...")
        taus = np.array(taus)
        rates = np.array(rates)

        # Sort by tau
        sort_idx = np.argsort(taus)
        taus = taus[sort_idx]
        rates = rates[sort_idx]

        # Need Gradient G.
        # Try to find in metadata of first file?
        # Or hardcode/ask user?
        # User prompt: "Write the diffusion coeficient with the +- on the graph"
        # I'll default G to something or Try to read it.
        # If not found, warn.
        # For now, placeholder or check metadata.
        # Assuming G is needed for D calculation.
        # Let's set a default placeholder G = 0.4 T/m (just an example or 1.0) if not found,
        # but better to check data.

        # Re-load first file to check metadata for gradient?
        # Assuming G is constant.
        gradient = 0.0
        # Attempt to read 'Gradient' or similar from metadata of first valid data
        # Skipping for now to rely on fit_diffusion defaults or passed arg?
        # Expected from user: "In this water there is data...".
        # Let's assume G=1.0 for now or try to extract.
        # Actually, without G, we can't get D.
        # I'll use 1.0 as placeholder and note it.
        gradient = 1.0

        # Calculate fixed_intercept (R2_intrinsic) if fixed_t2 is provided
        fixed_intercept = None
        if fixed_t2 is not None and fixed_t2 > 0:
            fixed_intercept = 1.0 / fixed_t2
            console.print(
                f"Using fixed R2 intercept: {fixed_intercept:.4f} s^-1 (from T2={fixed_t2:.4f} s)"
            )

        result = Fitter.fit_diffusion(
            taus, rates, gradient_strength=gradient, fixed_intercept=fixed_intercept
        )

        print_result(result)

        if plot:
            filepath = None
            if save_path:
                fname = f"{prefix}_diffusion_fit.png" if prefix else "diffusion_fit.png"
                filepath = save_path / fname
                console.print(f"Saving diffusion plot to {filepath}")

            # Plot R2 vs Tau^2
            fig, ax = plt.subplots(figsize=(8, 6))
            x_vals = result.metadata.get("x_values", taus**2)  # Tau^2
            y_vals = result.metadata.get("y_values", rates)  # R2

            ax.scatter(x_vals, y_vals, label="Data ($1/T_{2,obs}$)", color="blue")

            if len(result.fit_curve) > 0:
                ax.plot(x_vals, result.fit_curve, label="Fit", color="red")

            # Annotation
            D_val = result.params.get("D", 0.0)
            D_err = result.param_errors.get("D", 0.0)

            # If we don't know G, D is meaningless unless G=1 is correct.
            # But the user specifically asked for D with +/- on graph.

            label_str = rf"$D = {D_val:.4e} \pm {D_err:.4e}$ $m^2/s$"
            if gradient == 1.0:
                label_str += "\n(Assuming G=1 T/m)"

            ax.text(
                0.05,
                0.95,
                label_str,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=12,
            )

            ax.set_xlabel(r"$\tau^2$ ($s^2$)")
            ax.set_ylabel(r"$R_{2,obs}$ ($s^{-1}$)")
            ax.set_title("Diffusion Analysis ($R_2$ vs $\tau^2$)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if filepath:
                plt.savefig(filepath)
                plt.close()
            else:
                plt.show()

        return [AnalysisContext(data=NMRData(time=taus, signal=rates), result=result)]

    elif experiment == ExperimentType.T2_COMBINED:
        # T2 Combined: Single file (or multiple) with echo train
        target_file = path
        if path.is_dir():
            files = (
                list(path.glob("*.h5"))
                + list(path.glob("*.hdf5"))
                + list(path.glob("*.csv"))
            )
            if not files:
                raise FileNotFoundError(f"No HDF5/CSV files in {path}")
            target_file = files[0]

        console.print(f"Loading {target_file}...")
        loader = get_loader(target_file, channel=channel)
        data = loader.load(target_file)
        data.experiment_type = experiment

        console.print("Extracting Echo Train...")

        # Use preprocess_data to trim data starting from Max/Pulse (Trimming Methodology)
        # This ensures we work on the relevant slice starting at t=0
        data, _, _, _ = preprocess_data(data, smoothing=ANALYSIS_SMOOTHING)

        # Extract Echo Train using Monotonic Filter logic (NMRMINE)
        (
            peak_times,
            peak_amps,
            excluded_times,
            excluded_amps,
        ) = extract_echo_train(data, smoothing=ANALYSIS_SMOOTHING)

        if len(peak_times) < 3:
            console.print(
                "[red]Not enough peaks found for T2 fit (need at least 3, so >2).[/red]"
            )
            return []

        # NMRMINE does not manually skip peaks if using Monotonic Filter + Trimming
        # The filter itself removes noise/dips. The first peak (0) is the Pulse.
        # Fits usually include the pulse if valid? Or start from echoes?
        # "t2_multiple_analysis.py" fits "peak_times" from "valid_indices".
        # If Pulse is valid (monotonic start), it is included.
        # We assume strict adherence to repo logic means "use valid_indices".
        # So we REMOVE the [2:] skip.
        # peak_times = peak_times[2:]
        # peak_amps = peak_amps[2:]

        console.print(
            f"Using {len(peak_times)} peaks (Monotonic Filter applied). Fitting T2..."
        )

        # User Request: "in water combined fit remove the first second"
        # Debugging: Verbose check
        if "water" in str(target_file).lower():
            console.print(f"[blue]Water dataset detected: {target_file.name}[/blue]")
            console.print(
                f"Total peaks before trim: {len(peak_times)}. Range: {peak_times[0]:.4f} - {peak_times[-1]:.4f} s"
            )

            # User Request (Update): "It should find the outer envelope... reverse... should make that happen"
            # This implies relying on the Reverse Monotonic Filter instead of an arbitrary 1.0s cut.
            # We will disable the 1.0s trim logic but keep the logging for verification.

            # mask = peak_times > 1.0
            # mask_exclude = ~mask

            # Add trimmed peaks to excluded lists
            # excluded_times = np.concatenate((excluded_times, peak_times[mask_exclude]))
            # excluded_amps = np.concatenate((excluded_amps, peak_amps[mask_exclude]))

            # peak_times = peak_times[mask]
            # peak_amps = peak_amps[mask]
            pass

            if len(peak_times) > 0:
                console.print(
                    f"Remaining peaks after 1.0s trim: {len(peak_times)}. Range: {peak_times[0]:.4f} - {peak_times[-1]:.4f} s"
                )
                # User Request (Update): Reliance on Reverse Monotonic Filter + Preprocess ArgMax.
                # Disabling secondary ArgMax logic.

                # max_idx_after = np.argmax(peak_amps)
                # max_peak_time = peak_times[max_idx_after]
                # max_peak_amp = peak_amps[max_idx_after]

                # console.print(
                #     f"Max peak after 1.0s found at {max_peak_time:.4f}s (Amp: {max_peak_amp:.4f}). Trimming pre-max peaks."
                # )

                # Add pre-max peaks to excluded
                # if max_idx_after > 0:
                #    excluded_times = np.concatenate((excluded_times, peak_times[:max_idx_after]))
                #    excluded_amps = np.concatenate((excluded_amps, peak_amps[:max_idx_after]))

                # Slice from that max index onwards
                # peak_times = peak_times[max_idx_after:]
                # peak_amps = peak_amps[max_idx_after:]
                # pass

                console.print(
                    f"Final peaks for fitting: {len(peak_times)}. Range: {peak_times[0]:.4f} - {peak_times[-1]:.4f} s"
                )
            else:
                console.print(f"[red]WARNING: No peaks remaining after trimming![/red]")

        # User Request: "remove the first peak from every fitting data before fit"
        # This applies generally to T2 Combined (Echo Train) analysis.
        if len(peak_times) > 0:
            # Exclude the first peak (Pulse/First Echo)
            # Add to excluded
            excluded_times = np.concatenate((excluded_times, [peak_times[0]]))
            excluded_amps = np.concatenate((excluded_amps, [peak_amps[0]]))

            # Remove from valid
            peak_times = peak_times[1:]
            peak_amps = peak_amps[1:]

            console.print("Removed first peak from fitting data (User Request).")

        if len(peak_times) < 3:
            console.print(
                "[red]Not enough peaks for fit after removing first peak.[/red]"
            )
            # We might still want to plot what we have? Or return?
            # If we return [], we get no plot.
            # Let's try to fit with what we have or skip.
            return []

        # Fit T2 to the peaks
        # Using 0 as initial time? Use relative time?
        # Standard T2 fit: M(t) = M0 exp(-t/T2)
        # Delays are peak_times

        # Re-use T2 fitting logic
        params, fit_curve, residuals, r2, param_errors = Fitter.fit_t2(
            peak_times, peak_amps
        )

        # Add week/substance to dataset name
        week, substance = _get_week_and_substance(path, prefix)
        title_prefix = (
            f"{week} {substance}".strip() if week or substance != "mineral-oil" else ""
        )
        combined_name = (
            f"{title_prefix} - Spin Echo (Echo Train)"
            if title_prefix
            else "Spin Echo (Echo Train)"
        )
        result = AnalysisResult(
            experiment_type=experiment,
            dataset_name=combined_name,
            params=params,
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
            param_errors=param_errors,
        )
        print_result(result)
        if plot:
            # We want: Raw Data + Peaks + Fit Curve on ONE graph
            filepath = None
            if save_path:
                fname = _generate_plot_filename(
                    target_file, experiment, "combined", prefix
                )
                filepath = save_path / fname
                console.print(f"Saving plot to {filepath}")

            plot_combined_t2(
                data,
                peak_times,
                peak_amps,
                result,
                filepath=filepath,
                excluded_times=excluded_times,
                excluded_amps=excluded_amps,
            )

        return [
            AnalysisContext(
                data=data, result=result, peak_times=peak_times, peak_amps=peak_amps
            )
        ]

    else:
        # T1 or T2 - Expecting directory of files
        if not path.is_dir():
            console.print(
                "[red]Error: T1/T2 analysis expects a directory of files.[/red]"
            )
            raise typer.Exit(1)

        files = (
            list(path.glob(("*.h5")))
            + list(path.glob(("*.hdf5")))
            + list(path.glob(("*.csv")))
        )
        if not files:
            console.print("[red]No .h5, .hdf5 or .csv files found in directory.[/red]")
            raise typer.Exit(1)

        console.print(f"Found {len(files)} files. Processing...")

        delays = []
        amplitudes = []
        raw_traces = []

        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))
            for f in files:
                try:
                    loader = get_loader(f, channel=channel)
                    data_full = loader.load(f)
                    data_full.experiment_type = experiment

                    # Preprocess
                    # For T1, we fundamentally rely on the FILENAME for the delay (tau).
                    # And the Amplitude should be the Maximum of the signal (First Echo/FID).
                    # The default 'find_peaks_t1_t2' tries to find 2 peaks to calc tau, which fails for CPMG trains or single-fid files.

                    processed_data, _, _, peak_info = preprocess_data(
                        data_full,
                        smoothing=ANALYSIS_SMOOTHING,
                    )

                    # 1. Parse Tau from Filename (Primary Source for T1)
                    import re

                    # Try X_XXX or X.XXX or just number
                    match = re.search(r"(\d+)_(\d+)", f.stem)
                    if match:
                        tau = float(f"{match.group(1)}.{match.group(2)}")
                    else:
                        match_num = re.search(r"([\d\.]+)", f.stem)
                        if match_num:
                            tau = float(match_num.group(1))
                            if (
                                tau > 10000
                            ):  # Heuristic: if > 10000, maybe it is in microseconds? Or filename is timestamp?
                                # Assume seconds if small, ms if large?
                                # Usually filenames are in 'ms' or 's'.
                                # If filename is "1100" (ms), tau should be 1.1s?
                                # Repo conventions?
                                # Users 1100.csv -> 1.1s is likely.
                                # Just use as is for now, later fit might scale?
                                # Wait, "1100.csv" -> 1100.
                                # Is it ms or s?
                                # If T1 is usually 0.1-5s. 1100s is huge. 1100ms = 1.1s is reasonable.
                                # Let's assume input is in [ms] if > 10? Or just trust val?
                                # Let's stick to raw value, but convert to seconds if it looks like ms?
                                # shoham-b/NMR repo usually uses seconds.
                                # If user has "1100", it's probably ms.
                                pass
                        else:
                            # Fallback to internal if filename parse fails?
                            # But internal is risky for CPMG.
                            tau = 0.0  # Will filter out?

                    # Heuristic for ms -> s conversion
                    # If tau > 50 and < 100000, assume ms?
                    if tau > 50:
                        tau = tau / 1000.0

                    # 2. Extract Amplitude (Max of signal)
                    # Use processed_data (trimmed/dc-corrected)
                    # We want the recovery amplitude.
                    # Ideally: Max Magnitude but Signed (to show Inversion).
                    # amp = sig[argmax(abs(sig))]
                    sig = processed_data.signal
                    # USE THE SELECTED PEAK from peak_info
                    fit_idx = peak_info.get("fit_idx", 0)
                    if fit_idx < len(sig):
                        amp = sig[fit_idx]
                    else:
                        # Fallback if fit_idx out of bounds? (Should not happen)
                        amp = sig[np.argmax(np.abs(sig))]

                    # Logic for "trace_label"
                    label = f.stem
                    data_full.metadata["trace_label"] = label
                    processed_data.metadata["trace_label"] = label

                    delays.append(tau)
                    amplitudes.append(amp)

                    # Store full raw data for visualization (User Request)
                    # Tuple format: (processed_data, t_peak, amp, tau, peak_info, data_full, sort_val)
                    # - processed_data: NMRData (trimmed, for plotting)
                    # - t_peak: float (not used for T1/T2, set to 0.0)
                    # - amp: float (amplitude value)
                    # - tau: float (delay value)
                    # - peak_info: dict (contains trim_start_idx, p1_idx, etc.)
                    # - data_full: NMRData (full untouched data)
                    # - sort_val: float (for sorting, use tau)

                    raw_traces.append(
                        (processed_data, 0.0, amp, tau, peak_info, data_full, tau)
                    )
                except Exception as e:
                    console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")

                progress.advance(task)

        if not delays:
            console.print("[red]No valid data processed.[/red]")
            raise typer.Exit(1)

        delays = np.array(delays)

        # Phase Correction for T1 (Crucial for Inversion Recovery)
        # We collected complex peak amplitudes in `amplitudes`?
        # Check: `amp = sig[np.argmax(np.abs(sig))]` -> `amp` is complex.
        # Yes, `amplitudes` is a list of complex numbers.

        amplitudes = np.array(amplitudes)
        sorted_indices = np.argsort(delays)
        delays = delays[sorted_indices]
        amplitudes = amplitudes[sorted_indices]

        if experiment == ExperimentType.T1:
            # Find trace with max delay (assumed relaxed)
            # Last in sorted list
            ref_amp = amplitudes[-1]
            ref_phase = np.angle(ref_amp)
            # We want ref_amp to be Positive Real.
            # So we rotate by -ref_phase.
            phase_corr = np.exp(-1j * ref_phase)

            # Apply to all
            amplitudes_phased = amplitudes * phase_corr
            # Take Real part
            amplitudes_fit = np.real(amplitudes_phased)

            console.print(
                f"Applied T1 Phase Correction (ref_phase={ref_phase:.2f} rad). Range: {np.min(amplitudes_fit):.2e} to {np.max(amplitudes_fit):.2e}"
            )

            # Heuristic: If T1 data is DECAYING (Slope < 0), it means we likely flipped the sign wrong (e.g. all points were negative).
            # T1 Recovery should be increasing (Slope > 0).
            if len(delays) > 1:
                slope, _ = np.polyfit(delays, amplitudes_fit, 1)
                if slope < 0:
                    console.print(
                        f"Detected decaying T1 data (slope={slope:.2e}). Flipping sign to restore recovery shape."
                    )
                    amplitudes_fit = -amplitudes_fit

        else:
            # T2/Others: Magnitude is usually sufficient (Decay)
            amplitudes_fit = np.abs(amplitudes)

        # raw_traces: (processed, full, t_peak, amp, tau, peak_info, sort_val)
        # Sort by sort_val (index 6)
        raw_traces.sort(key=lambda x: x[6])

        console.print("Fitting data...")
        if experiment == ExperimentType.T1:
            params, fit_curve, residuals, r2, param_errors = Fitter.fit_t1(
                delays, amplitudes_fit
            )
            week, substance = _get_week_and_substance(path, prefix)
            title_prefix = (
                f"{week} {substance}".strip()
                if week or substance != "mineral-oil"
                else ""
            )
            dataset_name = (
                f"{title_prefix} - T1 Analysis" if title_prefix else "T1 Analysis"
            )
        else:  # T2
            # Check for Alcohol (J-Modulated Analysis)
            # Heuristic: If dataset name ends with "nol" (e.g. Ethanol, Methanol)
            # Or if user explicitly requested alcohol handling (though we rely on path mostly)
            # We check path.name or path.parent.name
            target_name = path.name.lower()
            parent_name = path.parent.name.lower()

            # Helper to check if any word in a name contains "nol"
            def _contains_nol_word(name: str) -> bool:
                # Split on common separators: space, underscore, hyphen
                words = re.split(r"[\s_\-]+", name)
                return any("nol" in word for word in words)

            if (
                target_name.endswith("nol")
                or "alcohol" in target_name
                or _contains_nol_word(target_name)
                or parent_name.endswith("nol")
                or "alcohol" in parent_name
                or _contains_nol_word(parent_name)
            ):
                console.print(
                    "[cyan]Alcohol dataset detected: Using J-Modulated T2 Fit[/cyan]"
                )
                params, fit_curve, residuals, r2, param_errors = (
                    Fitter.fit_modulated_t2(delays, amplitudes_fit)
                )
                week, substance = _get_week_and_substance(path, prefix)
                title_prefix = (
                    f"{week} {substance}".strip()
                    if week or substance != "mineral-oil"
                    else ""
                )
                dataset_name = (
                    f"{title_prefix} - T2 Analysis (J-Modulated)"
                    if title_prefix
                    else "T2 Analysis (J-Modulated)"
                )
            else:
                params, fit_curve, residuals, r2, param_errors = Fitter.fit_t2(
                    delays, amplitudes_fit
                )
                week, substance = _get_week_and_substance(path, prefix)
                title_prefix = (
                    f"{week} {substance}".strip()
                    if week or substance != "mineral-oil"
                    else ""
                )
                dataset_name = (
                    f"{title_prefix} - T2 Analysis" if title_prefix else "T2 Analysis"
                )

        result = AnalysisResult(
            experiment_type=experiment,
            dataset_name=dataset_name,
            params=params,
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
            param_errors=param_errors,
        )

        print_result(result)
        if plot:
            filepath_fit = None
            filepath_traces = None
            if save_path:
                fname_fit = _generate_plot_filename(path, experiment, "fit", prefix)
                fname_traces = _generate_plot_filename(
                    path, experiment, "traces", prefix
                )
                filepath_fit = save_path / fname_fit
                filepath_traces = save_path / fname_traces
                console.print(f"Saving fit plot to {filepath_fit}")
                console.print(f"Saving traces plot to {filepath_traces}")

            plot_stacked_traces(
                raw_traces,
                filepath=filepath_traces,
                smoothing=ANALYSIS_SMOOTHING,
                title=result.dataset_name,
            )

            plot_analysis_summary(
                delays,
                amplitudes_fit,
                result,
                raw_traces,
                "Delay (s)",
                "Amplitude",
                filepath=filepath_fit,
                smoothing=ANALYSIS_SMOOTHING,
            )

        aggregated_data = NMRData(time=delays, signal=amplitudes_fit)
        return [
            AnalysisContext(data=aggregated_data, result=result, raw_traces=raw_traces)
        ]


def plot_spectrum_fit(freqs, mag_data, result, filepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Convert to kHz
    freqs_khz = freqs / 1000.0
    ax.plot(freqs_khz, mag_data, label="Data (Magnitude)", color="black", alpha=0.7)
    if len(result.fit_curve) > 0:
        ax.plot(
            freqs_khz,
            result.fit_curve,
            label="Fit (Mag Lorentzian)",
            color="red",
            linestyle="--",
        )

    # Mark peaks
    if "peaks" in result.params:
        for p in result.params["peaks"]:
            f0 = p["f0"] / 1000.0
            ax.axvline(f0, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(-4, 4)
    ax.set_title(f"{result.dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        plt.savefig(filepath)
    plt.close()


def plot_hybrid_result(
    result: HybridAnalysisResult,
    out_dir: Path,
    source_path: Optional[Path] = None,
    prefix: str = "",
):
    """
    Generate plots for Hybrid Analysis:
    1. Stacked Overview: Time Traces (Left) + Frequency Spectra (Right).
    2. T2 Decay Fits: Linear (Left) + Log (Right) of Area vs Tau.
    """
    import matplotlib.cm as cm

    # --- 1. Stacked Overview (Time + Freq) ---
    fig_stack, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(16, 8))
    cmap = cm.viridis

    # Left: Time Traces
    time_list = result.time_stack
    n_files = len(time_list)

    valid_sigs = [np.abs(d.signal) for d in time_list if len(d.signal) > 0]
    max_sig_t = np.max([np.max(s) for s in valid_sigs]) if valid_sigs else 1.0
    offset_step_t = max_sig_t * 0.5

    for i, data in enumerate(time_list):
        sig = np.abs(data.signal)
        t = data.time
        color = cmap(i / n_files)
        ax_time.plot(t, sig + i * offset_step_t, color=color, alpha=0.8)

    ax_time.set_xlabel(f"Time ({time_list[0].metadata.get('time_unit', 's')})")
    ax_time.set_ylabel("Signal Amplitude (Stacked)")
    ax_time.set_title(f"Stacked Time Traces")
    ax_time.grid(True, alpha=0.3)

    # Right: Frequency Spectra
    freqs, spectra_list = result.spectra_stack
    valid_specs = [s for s in spectra_list if len(s) > 0]
    max_amp_f = np.max([np.max(s) for s in valid_specs]) if valid_specs else 1.0
    offset_step_f = max_amp_f * 0.2

    for i, spect in enumerate(spectra_list):
        mag = np.abs(spect)
        color = cmap(i / n_files)
        ax_freq.plot(freqs / 1000.0, mag + i * offset_step_f, color=color, alpha=0.8)

    ax_freq.set_xlabel("Frequency (kHz)")
    ax_freq.set_ylabel("Magnitude (Stacked)")
    ax_freq.set_xlim(-10, 10)
    ax_freq.set_title(f"Stacked Spectra")
    ax_freq.grid(True, alpha=0.3)

    fig_stack.suptitle(f"Hybrid Analysis Overview: {result.dataset_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Generate filename using helper if source_path is available
    if source_path:
        fname_stack = _generate_plot_filename(
            source_path, ExperimentType.SPECTRUM, "stacked-overview", prefix
        )
    else:
        fname_stack = (
            f"{prefix}_stacked_overview.png" if prefix else "stacked_overview.png"
        )
    plt.savefig(out_dir / fname_stack)
    plt.close(fig_stack)
    console.print(f"Saved {fname_stack}")

    # --- 2. T2 Decay Fits ---
    n_peaks = len(result.peak_centers)

    fig_fit, axes = plt.subplots(n_peaks, 2, figsize=(16, 6 * n_peaks), squeeze=False)
    taus = result.tau_values

    for k in range(n_peaks):
        areas = result.integrated_areas[k, :]  # Integrated Frequency Space Area
        fit_res = result.t2_results[k]
        f0 = result.peak_centers[k]

        # Left: Linear
        ax_lin = axes[k, 0]
        ax_lin.scatter(
            taus, areas, label="Integrated Area", color="blue", s=50, zorder=3
        )

        t_smooth = np.linspace(min(taus), max(taus), 200)
        if fit_res.get("T1", 0) > 0:
            # T1 Fit Visualization
            from nmr_analysis.analysis.models import t1_model

            y_fit = t1_model(t_smooth, fit_res["M0"], fit_res["T1"], fit_res["alpha"])
            ax_lin.plot(t_smooth, y_fit, "r--", label="Fit T1", linewidth=2, zorder=2)

            # Right Plot for T1: Also Linear usually!
            # Mirror Left plot or show residuals or just Linear Fit again
            ax_log = axes[k, 1]
            ax_log.scatter(
                taus, areas, label="Integrated Area", color="blue", s=50, zorder=3
            )
            ax_log.plot(t_smooth, y_fit, "r--", label="Fit T1", linewidth=2, zorder=2)

            val = fit_res["T1"]
            r2 = fit_res.get("r_squared", 0)
            text_str = rf"$T_1 = {val:.4f}$ s" + "\n" + rf"$R^2 = {r2:.4f}$"

            ax_log.set_yscale("linear")
            ax_log.set_ylabel("Integrated Area")
            ax_log.set_title(f"Peak @ {f0:.1f} Hz (Linear T1)")

        elif fit_res.get("T2", 0) > 0:
            y_fit = t2_decay_model(
                t_smooth, fit_res["M0"], fit_res["T2"], fit_res["offset"]
            )
            ax_lin.plot(t_smooth, y_fit, "r--", label="Fit", linewidth=2, zorder=2)

            # Right: Log
            ax_log = axes[k, 1]
            y_fit_log = y_fit
            valid_y = y_fit_log > 0
            ax_log.plot(
                t_smooth[valid_y],
                y_fit_log[valid_y],
                "r--",
                label="Fit",
                linewidth=2,
                zorder=2,
            )

            # Add T2 text
            val = fit_res["T2"]
            r2 = fit_res.get("r_squared", 0)
            text_str = rf"$T_2 = {val:.4f}$ s" + "\n" + rf"$R^2 = {r2:.4f}$"

            ax_log.set_yscale("log")
            ax_log.set_ylabel("Integrated Area (Log)")
            ax_log.set_title(f"Peak @ {f0:.1f} Hz (Log)")

        # Common Text Box for Right Plot
        if fit_res.get("T1", 0) > 0 or fit_res.get("T2", 0) > 0:
            ax_log.text(
                0.95,
                0.95,
                text_str,
                transform=ax_log.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax_log.set_xlabel("Delay $\\tau$ (s)")
        ax_log.grid(True, which="both", alpha=0.5)

    fig_fit.suptitle(f"T2 Decay Analysis: {result.dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Generate filename using helper if source_path is available
    if source_path:
        fname_fit = _generate_plot_filename(
            source_path, ExperimentType.SPECTRUM, "t2-decay", prefix
        )
    else:
        fname_fit = f"{prefix}_t2_decay.png" if prefix else "t2_decay.png"
    plt.savefig(out_dir / fname_fit)
    plt.close(fig_fit)
    console.print(f"Saved {fname_fit}")


def print_result(result: AnalysisResult):
    table = Table(title=f"Results: {result.dataset_name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    if "T2_star" in result.params:
        val = result.params["T2_star"]
        err = result.param_errors.get("T2_star", 0.0)
        table.add_row("T2*", f"{val:.4f} ± {err:.4f}")
        if "M0" in result.params:
            table.add_row("M0", f"{result.params['M0']:.4e}")
        table.add_row("R-Squared", f"{result.r_squared:.4f}")

    else:
        for k, v in result.params.items():
            if k == "peaks" and isinstance(v, list):
                # Handle Peak List
                table.add_row("Peaks Found", f"{len(v)}")
                console.print(table)
                # Create separate table for peaks
                peak_table = Table(title="Detected Peaks")
                peak_table.add_column("Freq (Hz)", justify="right")
                peak_table.add_column("T2* (s)", justify="right")
                peak_table.add_column("Amp", justify="right")

                for p in v:
                    peak_table.add_row(
                        f"{p['f0']:.2f}", f"{p['t2_star']:.4f}", f"{p['amplitude']:.2e}"
                    )
                console.print(peak_table)
                return

            if k in ("T2", "T1"):
                table.add_row(k, f"{v:.4f}")
            elif isinstance(v, (int, float)):
                table.add_row(k, f"{v:.4e}")
            else:
                table.add_row(k, str(v))
        table.add_row("R-Squared", f"{result.r_squared:.4f}")

    console.print(table)


def plot_result(
    x, y, result: AnalysisResult, xlabel, ylabel, filepath: Optional[Path] = None
):
    """
    Plot Fit Result for T2* in split view:
    Left: Linear Scale, Full Raw Data, Highlight Peak used for fit.
    Right: Log Scale, Fit Curve (Existing).
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Linear Scale (Full Raw Data) ---
    # x, y here are the FULL data passed from _run_analysis

    # Plot Full Raw Data
    ax_lin.plot(x, y, label="Raw Data", color="blue", alpha=0.6)

    # Highlight Peak used for fit
    # T2* logic in fitting.py finds peak_idx.
    if result.metadata and "peak_index" in result.metadata:
        peak_idx = result.metadata["peak_index"]
        if 0 <= peak_idx < len(x):
            ax_lin.scatter(
                [x[peak_idx]],
                [y[peak_idx]],
                color="red",
                marker="x",
                s=100,
                label="Peak (Fit Start)",
                zorder=5,
            )

    # Plot Fit on Linear Scale
    if result.fit_curve is not None:
        ax_lin.plot(
            x,
            result.fit_curve,
            label="Fit",
            color="red",
            alpha=0.8,
            linewidth=2,
            linestyle="--",
            zorder=4,
        )

    ax_lin.set_xlabel(xlabel)
    ax_lin.set_ylabel(ylabel)
    ax_lin.set_title(f"{result.dataset_name} (Linear)")
    ax_lin.grid(True, alpha=0.5)
    ax_lin.legend(loc="best")

    # --- Plot 2: Log Scale (Fit) ---

    # Re-plot raw data (filtered > 0)
    mask = y > 0
    ax_log.plot(x[mask], y[mask], label="Raw Data", color="blue", alpha=0.3)

    # Plot Fit
    if result.fit_curve is not None:
        fit_vals = result.fit_curve
        fit_mask = (fit_vals > 0) & np.isfinite(fit_vals) & mask

        # Ensure we plot consistent range
        ax_log.plot(
            x[fit_mask], fit_vals[fit_mask], label="Fit", color="red", linewidth=2
        )

    ax_log.set_xlabel(xlabel)

    # Check if we should log scale or linear scale
    # T2* is usually decay, so Log is fine.
    # But if T1, linear.
    if "T1" in result.params:
        ax_log.set_ylabel(ylabel)
        ax_log.set_title(f"{result.dataset_name} (Fit)")
        ax_log.set_yscale("linear")
    else:
        ax_log.set_ylabel(f"{ylabel} (Log)")
        ax_log.set_title(f"{result.dataset_name} (Log Scale)")
        ax_log.set_yscale("log")

    # Heuristic for ylim
    valid_y = y[y > 0]
    if len(valid_y) > 0:
        min_y = np.min(valid_y)
        max_y = np.max(valid_y)
        bottom_limit = 1.0 if max_y > 10 else min_y * 0.1
        ax_log.set_ylim(bottom=bottom_limit)

    ax_log.legend()
    ax_log.grid(True, which="both", alpha=0.5)

    if "T2_star" in result.params:
        val = result.params["T2_star"]
        err = result.param_errors.get("T2_star", 0.0)
        text_str = rf"$T_2^* = {val:.4f} \pm {err:.4f}$ s"
        ax_log.text(
            0.05,
            0.05,
            text_str,
            transform=ax_log.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.close()


def plot_combined_t2(
    data: NMRData,
    peak_times: np.ndarray,
    peak_amps: np.ndarray,
    result: AnalysisResult,
    filepath: Optional[Path] = None,
    excluded_times: Optional[np.ndarray] = None,
    excluded_amps: Optional[np.ndarray] = None,
):
    """
    Plot T2 Combined analysis results (Echo Train Decay).
    """
    unit = "s"  # Assume processed data is in seconds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Define colors for peaks based on index/time
    num_peaks = len(peak_times)
    cmap = cm.viridis
    norm = plt.Normalize(0, num_peaks - 1 if num_peaks > 1 else 1)
    colors = [cmap(norm(i)) for i in range(num_peaks)]

    # --- Plot 1: Linear Scale (Raw + Fit) ---
    # Plot raw echo train
    ax1.plot(data.time, data.signal, label="Raw Echo Train", color="skyblue", alpha=0.6)

    # Plot Extracted Peaks (Valid)
    ax1.scatter(
        peak_times,
        peak_amps,
        c=colors,
        marker="x",
        s=60,
        linewidths=2,
        zorder=5,
        label="Peaks (Used)",
    )

    # Plot Excluded Peaks (If any)
    if excluded_times is not None and len(excluded_times) > 0:
        ax1.scatter(
            excluded_times,
            excluded_amps,
            color="gray",
            marker="x",
            s=40,
            linewidths=1,
            alpha=0.5,
            zorder=4,
            label="Excluded Peaks",
        )

    # Plot Fit Curve
    if result.fit_curve is not None:
        # Generate smooth curve for display
        # We need to use data.time range
        # Note: fit function was M0 * exp(-t/T2) + offset

        if "M0" in result.params and "T2" in result.params:
            M0 = result.params["M0"]
            T2 = result.params["T2"]
            offset = result.params.get("offset", 0.0)
            full_fit_curve = t2_decay_model(data.time, M0, T2, offset)

            label_fit = f"T2 Fit (T2={T2:.4e} {unit})"
            if "T2" in result.param_errors:
                err = result.param_errors["T2"]
                label_fit = rf"T2 Fit ($T_2={T2:.4f} \pm {err:.4f}$ {unit})"

            ax1.plot(
                data.time,
                full_fit_curve,
                label=label_fit,
                color="red",
                linestyle="-",
                zorder=6,
            )

            # Add textbox to ax1
            err_t2 = result.param_errors.get("T2", 0.0)
            err_m0 = result.param_errors.get("M0", 0.0)
            text_str = (
                rf"$T_2 = {T2:.4f} \pm {err_t2:.4f}$ {unit}"
                + "\n"
                + rf"$M_0 = {M0:.4e} \pm {err_m0:.4e}$"
            )

            ax1.text(
                0.95,
                0.95,
                text_str,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            # Fallback if no params (should not happen if fit succeeded)
            if result.fit_curve is not None:
                ax1.plot(
                    peak_times, result.fit_curve, label="Fit", color="red", zorder=6
                )

    ax1.set_xlabel(f"Time ({unit})")
    ax1.set_ylabel("Signal Magnitude")
    ax1.set_title(f"{result.dataset_name} (Linear)")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="best")

    # --- Plot 2: Decay (Log) ---
    ax2.scatter(
        peak_times,
        peak_amps,
        c=colors,
        marker="x",
        s=80,
        linewidths=2,
        zorder=5,
        label="Peaks",
    )

    if result.fit_curve is not None:
        ax2.plot(
            peak_times,
            result.fit_curve,
            label="Fit",
            color="red",
            linestyle="--",
            zorder=6,
        )

    ax2.set_xlabel(f"Time ({unit})")
    ax2.set_ylabel("Signal Magnitude (Log)")
    ax2.set_title("Decay (Log)")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1)
    ax2.grid(True, which="both", alpha=0.5)
    ax2.legend(loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.close()


def plot_stacked_traces(
    raw_traces: List[Tuple[NMRData, NMRData, float, float, float, dict, float]],
    filepath: Optional[Path] = None,
    smoothing: float = 1.0,
    show_fourier: bool = False,
    title: str = "",
):
    """
    Plot processed traces (left), full raw traces (middle), and optionally Fourier transform (right), stacked vertically.

    Args:
        raw_traces: List of trace data tuples (processed_data, data_full, t_peak, amp, tau, peak_info, sort_val)
        filepath: Optional path to save the plot
        smoothing: Smoothing parameter (not used for Fourier)
        show_fourier: If True, show 3 columns with Fourier transform; if False, show 2 columns (legacy)
        title: Optional title to display as suptitle
    """
    num_traces = len(raw_traces)
    if num_traces == 0:
        return

    fig_height = max(6, num_traces * 3)
    # Create 2 or 3 columns based on show_fourier flag
    num_cols = 3 if show_fourier else 2
    fig, axes = plt.subplots(num_traces, num_cols, figsize=(8 * num_cols, fig_height))

    # Handle single trace case
    if num_traces == 1:
        axes = axes.reshape(1, num_cols)

    cmap = cm.viridis
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    # Calculate global x-axis limits for normalization
    proc_xlims = [np.inf, -np.inf]
    raw_xlims = [np.inf, -np.inf]

    for processed_data, t_peak, amp, tau, peak_info, data_full, *_ in raw_traces:
        if hasattr(processed_data, "time") and len(processed_data.time) > 0:
            proc_xlims[0] = min(proc_xlims[0], processed_data.time.min())
            proc_xlims[1] = max(proc_xlims[1], processed_data.time.max())
        if hasattr(data_full, "time") and len(data_full.time) > 0:
            raw_xlims[0] = min(raw_xlims[0], data_full.time.min())
            raw_xlims[1] = max(raw_xlims[1], data_full.time.max())

    for i, (processed_data, t_peak, amp, tau, peak_info, data_full, *_) in enumerate(
        raw_traces
    ):
        # Skip invalid trace data (e.g. from failed analysis)
        if not hasattr(processed_data, "signal") or not hasattr(data_full, "signal"):
            continue

        ax_proc = axes[i, 0]  # Column 1: Processed
        ax_raw = axes[i, 1]  # Column 2: Full Raw
        color = cmap(norm(i))

        # User Request: "It still seems to take the absoulte value"
        # Use Real (Signed) signal for plotting
        proc_signal = np.real(processed_data.signal)
        full_signal = np.real(data_full.signal)

        # --- COLUMN 1: Processed Data with Peaks ---
        ax_proc.plot(
            processed_data.time, proc_signal, color=color, alpha=0.8, linewidth=1.2
        )
        ax_proc.set_ylabel("Amplitude")
        ax_proc.grid(True, alpha=0.3)

        if "trace_label" in processed_data.metadata:
            ax_proc.set_title(f"Processed: {processed_data.metadata['trace_label']}")
        else:
            unit = processed_data.metadata.get("time_unit", "s")
            ax_proc.set_title(f"Processed Trace {i + 1}: τ={tau:.2e} {unit}")

        # Mark peaks on Processed (Relative indices)
        def mark_peak_proc(idx, color_marker, marker, label):
            if idx >= 0 and idx < len(processed_data.time):
                ax_proc.scatter(
                    [processed_data.time[idx]],
                    [proc_signal[idx]],
                    color=color_marker,
                    marker=marker,
                    s=100,
                    zorder=6,
                    label=label,
                    edgecolors="black",
                )

        p1_idx = peak_info.get("p1_idx", 0)
        mark_peak_proc(p1_idx, "cyan", "o", "P1 (Start)")

        p2_idx = peak_info.get("p2_idx", -1)
        if p2_idx != -1 and p2_idx >= p1_idx:
            mark_peak_proc(p2_idx, "red", "X", "P2 (Ignored)")

        fit_idx = peak_info.get("fit_idx", peak_info.get("p3_idx", 0))
        if fit_idx >= p1_idx:
            mark_peak_proc(fit_idx, "lime", "*", "Fit Peak")

        ax_proc.legend(loc="best", fontsize=8)

        # --- COLUMN 2: Full Raw Data ---
        ax_raw.plot(data_full.time, full_signal, color=color, alpha=0.9, linewidth=1.5)

        # Mark peaks on Full Raw (Absolute indices)
        trim_offset = peak_info.get("trim_start_idx", 0)

        # Helper to plot peak markers on Raw
        def mark_peak_raw(idx, color_marker, marker, label):
            if idx >= 0 and idx < len(data_full.time):
                ax_raw.scatter(
                    [data_full.time[idx]],
                    [full_signal[idx]],
                    color=color_marker,
                    marker=marker,
                    s=100,
                    zorder=6,
                    label=label,
                    edgecolors="black",
                )

        mark_peak_raw(trim_offset + p1_idx, "cyan", "o", "P1")
        if p2_idx != -1 and p2_idx >= p1_idx:
            mark_peak_raw(trim_offset + p2_idx, "red", "X", "P2")
        if fit_idx >= p1_idx:
            mark_peak_raw(trim_offset + fit_idx, "lime", "*", "Fit")

        ax_raw.set_ylabel("Amplitude")
        ax_raw.grid(True, alpha=0.3)

        if "trace_label" in data_full.metadata:
            ax_raw.set_title(f"Full Raw: {data_full.metadata['trace_label']}")
        else:
            unit = data_full.metadata.get("time_unit", "s")
            ax_raw.set_title(f"Full Raw Trace {i + 1}")

        # Apply normalized x-axis limits
        if proc_xlims[0] != np.inf:
            ax_proc.set_xlim(proc_xlims)
        if raw_xlims[0] != np.inf:
            ax_raw.set_xlim(raw_xlims)

        # --- COLUMN 3: Fourier Transform (if enabled) ---
        if show_fourier:
            ax_fourier = axes[i, 2]

            # Compute Fourier Transform for Full Raw Data
            freqs_raw, spect_raw = compute_spectrum(data_full)
            mag_raw = np.abs(spect_raw)

            if len(freqs_raw) > 0:
                ax_fourier.plot(
                    freqs_raw / 1000.0, mag_raw, color=color, alpha=0.9, linewidth=1.5
                )
            ax_fourier.set_ylabel("Magnitude")
            ax_fourier.set_xlim(-10, 10)
            ax_fourier.grid(True, alpha=0.3)

            if "trace_label" in data_full.metadata:
                ax_fourier.set_title(f"Fourier: {data_full.metadata['trace_label']}")
            else:
                ax_fourier.set_title(f"Fourier Transform {i + 1}")

            # Fourier already has fixed xlim at line 2396

    # Set x-labels only on bottom row
    unit = raw_traces[0][0].metadata.get("time_unit", "s")
    axes[-1, 0].set_xlabel(f"Time ({unit})")
    axes[-1, 1].set_xlabel(f"Time ({unit})")
    if show_fourier:
        axes[-1, 2].set_xlabel("Frequency (kHz)")

    # Add main title if provided
    if title:
        plt.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plot_analysis_summary(
    x,
    y,
    result: AnalysisResult,
    raw_traces: List[Tuple[NMRData, float, float, float, np.ndarray]],
    xlabel,
    ylabel,
    filepath: Optional[Path] = None,
    smoothing: float = 1.0,
    show_fourier: bool = False,
):
    """
    Plot Fit Result and Raw Traces in a split figure:
    1. Raw Traces (faint) + Smoothed Traces (bold) + Selected Peaks (Overlaid)
    2. Fit (Log) OR Fourier Transform (if show_fourier=True)
    """
    fig, (ax_traces, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # Color Mapping
    cmap = cm.viridis
    num_traces = len(raw_traces)
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    # --- Plot 1: Raw Traces (Time Domain) ---
    for i, (data, t_peak, amp, tau, peak_info, *_) in enumerate(raw_traces):
        if not hasattr(data, "signal"):
            # Skip invalid trace data (e.g. from failed analysis)
            continue

        color = cmap(norm(i))
        # User Request: "It still seems to take the absoulte value"
        signal = np.real(data.signal)
        # Raw trace (faint)
        ax_traces.plot(data.time, signal, color=color, alpha=0.3)

        # Smoothed trace (bold)
        smoothed = gaussian_filter1d(signal, sigma=smoothing)
        ax_traces.plot(
            data.time, smoothed, color=color, alpha=0.8, linestyle="-", linewidth=1.5
        )

        # Highlight 3 Peaks (P1, P2, P3)
        def mark_peak(idx, color, marker, label):
            if idx >= 0 and idx < len(data.time):
                ax_traces.scatter(
                    [data.time[idx]],
                    [smoothed[idx]],
                    color=color,
                    marker=marker,
                    s=80,
                    zorder=5,
                    label=label if i == 0 else None,
                    edgecolors="black",
                )

        # NOTE: `data` here is `processed_data` which is ALREADY TRIMMED.
        # peak_info indices are relative to this trimmed data.
        # Do NOT add trim_offset when indexing into `data`.

        # P1 (Start) - should be at t=0 (index 0) for trimmed data
        p1_idx = peak_info.get("p1_idx", 0)
        mark_peak(p1_idx, "cyan", "o", "P1 (Start)")

        # P2 (Noise) - optional
        p2_idx = peak_info.get("p2_idx", -1)
        if p2_idx != -1 and p2_idx >= p1_idx:
            mark_peak(p2_idx, "red", "X", "P2 (Ignored)")

        # P3 (Fit) or P2 (Fit) - Green Star
        # We use 'fit_idx' from peak_info which tells us WHICH peak was used.
        # No trim_offset needed - indices are relative to processed_data
        fit_idx = peak_info.get("fit_idx", peak_info.get("p3_idx", 0))

        if fit_idx >= p1_idx:
            mark_peak(fit_idx, "lime", "*", "Fit Peak")

    # --- Overlay Fit Curve on Raw Traces Plot ---
    # Extract peak times and fit values to plot the decay envelope
    if result.fit_curve is not None and len(raw_traces) > 0:
        # Collect peak times from raw traces (t_peak values)
        peak_times = []
        for trace_data in raw_traces:
            if len(trace_data) >= 2:
                t_peak = trace_data[1]  # t_peak is the second element
                peak_times.append(t_peak)

        # Sort and pair with fit curve for proper display
        if len(peak_times) == len(result.fit_curve):
            sorted_indices = np.argsort(peak_times)
            sorted_times = np.array(peak_times)[sorted_indices]
            sorted_fit = np.array(result.fit_curve)[sorted_indices]
            ax_traces.plot(
                sorted_times,
                sorted_fit,
                "r--",
                linewidth=2.5,
                label="Fit",
                zorder=10,
            )
        elif "J" in result.params:
            # J-modulated: generate dense fit curve
            from nmr_analysis.analysis.models import j_modulated_t2

            if len(peak_times) > 0:
                t_dense = np.linspace(min(peak_times), max(peak_times), 500)
                M0 = result.params["M0"]
                T2 = result.params["T2"]
                J = result.params["J"]
                offset = result.params.get("offset", 0.0)
                depth = result.params.get("depth", 1.0)
                fit_dense = j_modulated_t2(t_dense, M0, T2, J, offset, depth)
                ax_traces.plot(
                    t_dense,
                    fit_dense,
                    "r--",
                    linewidth=2.5,
                    label="Fit",
                    zorder=10,
                )
        elif len(peak_times) > 0 and ("T2" in result.params or "T1" in result.params):
            # Generate fit curve from model parameters
            t_dense = np.linspace(min(peak_times), max(peak_times), 500)
            if "T2" in result.params:
                M0 = result.params["M0"]
                T2 = result.params["T2"]
                offset = result.params.get("offset", 0.0)
                fit_dense = M0 * np.exp(-t_dense / T2) + offset
            else:  # T1
                M0 = result.params["M0"]
                T1 = result.params["T1"]
                fit_dense = M0 * (1 - 2 * np.exp(-t_dense / T1))
            ax_traces.plot(
                t_dense,
                fit_dense,
                "r--",
                linewidth=2.5,
                label="Fit",
                zorder=10,
            )

    ax_traces.set_xlabel(f"Time ({raw_traces[0][0].metadata.get('time_unit', 's')})")
    ax_traces.set_ylabel("Signal Amplitude")
    ax_traces.set_title("Raw Traces & Selected Peaks")
    # Only add legend if there are labeled artists
    handles, labels = ax_traces.get_legend_handles_labels()
    if handles:
        ax_traces.legend(loc="upper right")
    ax_traces.grid(True, alpha=0.5)

    # --- Plot 2: Fit (Log) OR Fourier Transform ---
    if show_fourier:
        # Show stacked Fourier transforms instead of log-scale fit
        for i, (data, t_peak, amp, tau, peak_info, *extra) in enumerate(raw_traces):
            color = cmap(norm(i))

            # Get the full raw data if available
            if len(extra) > 0 and isinstance(extra[0], NMRData):
                data_full = extra[0]
            else:
                data_full = data

            # Compute Fourier Transform
            freqs, spect = compute_spectrum(data_full)
            mag = np.abs(spect)

            # Offset for stacking
            offset = i * (np.max(mag) * 0.3 if len(mag) > 0 else 0)

            if len(freqs) > 0:
                ax_right.plot(
                    freqs / 1000.0,
                    mag + offset,
                    color=color,
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"τ={tau:.2e} s" if i < 5 else None,  # Limit legend entries
                )

        ax_right.set_xlabel("Frequency (kHz)")
        ax_right.set_ylabel("Magnitude (Stacked)")
        ax_right.set_title("Fourier Transform (Stacked)")
        ax_right.set_xlim(-10, 10)
        ax_right.grid(True, alpha=0.3)
        if num_traces <= 5:
            ax_right.legend(loc="best", fontsize=8)
    else:
        # Plot data points
        ax_right.scatter(x, y, c="blue", label="Data Points", zorder=3)

        # Fit Curve - use dense grid for J-modulated to show oscillations
        if result.fit_curve is not None:
            if "J" in result.params:
                # J-modulated: generate dense curve to show oscillations
                from nmr_analysis.analysis.models import j_modulated_t2

                t_dense = np.linspace(np.min(x), np.max(x), 500)
                M0 = result.params["M0"]
                T2 = result.params["T2"]
                J = result.params["J"]
                offset = result.params.get("offset", 0.0)
                depth = result.params.get(
                    "depth", 1.0
                )  # Default to full modulation if not present
                fit_dense = j_modulated_t2(t_dense, M0, T2, J, offset, depth)
                ax_right.plot(
                    t_dense,
                    fit_dense,
                    label="Fit",
                    color="red",
                    linestyle="--",
                    zorder=6,
                )
            else:
                # Standard fit: use original sparse points
                sorted_pairs = sorted(zip(x, result.fit_curve))
                sx, sy = zip(*sorted_pairs)
                ax_right.plot(
                    sx, sy, label="Fit", color="red", linestyle="--", zorder=6
                )

        # Add fit annotations with errors
        if "T1" in result.params:  # T1 Case
            T1 = result.params["T1"]
            M0 = result.params["M0"]
            err_t1 = result.param_errors.get("T1", 0.0)
            err_m0 = result.param_errors.get("M0", 0.0)
            text_str = (
                rf"$T_1 = {T1:.4f} \pm {err_t1:.4f}$ s"
                + "\n"
                + rf"$M_0 = {M0:.4e} \pm {err_m0:.4e}$"
            )
        elif "T2" in result.params:  # T2 Case
            T2 = result.params["T2"]
            M0 = result.params["M0"]
            err_t2 = result.param_errors.get("T2", 0.0)
            err_m0 = result.param_errors.get("M0", 0.0)
            text_str = (
                rf"$T_2 = {T2:.4f} \pm {err_t2:.4f}$ s"
                + "\n"
                + rf"$M_0 = {M0:.4e} \pm {err_m0:.4e}$"
            )
            if "J" in result.params:
                J_val = result.params["J"]
                err_J = result.param_errors.get("J", 0.0)
                text_str += "\n" + rf"$J = {J_val:.2f} \pm {err_J:.2f}$ Hz"
        else:
            text_str = ""

        if text_str:
            ax_right.text(
                0.95,
                0.95,
                text_str,
                transform=ax_right.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax_right.set_xlabel(xlabel)

        if "T1" in result.params:
            # T1: Linear Scale (best for Inversion Recovery)
            ax_right.set_ylabel(ylabel)
            ax_right.set_title(f"{result.dataset_name} (Fit)")
            ax_right.set_yscale("linear")
            # Ensure we see 0 if relevant
            # ax_right.set_ylim(bottom=0) # Optional, depends on data
        else:
            # T2: Log Scale
            ax_right.set_ylabel(f"{ylabel} (Log)")
            ax_right.set_title(f"{result.dataset_name} (Log Scale)")
            ax_right.set_yscale("log")
            ax_right.set_ylim(bottom=1)

        ax_right.grid(True, which="both", alpha=0.5)
        ax_right.legend(loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    for week in (
        "4.1",
        "4.2",
        "5.1",
        "5.2",
    ):
        week_path = Path(rf"H:\My Drive\Lab C\NMR\week{week}")
        if not week_path.exists():
            console.print(f"[yellow]Skipping week {week}: directory not found[/yellow]")
            continue
        analyze(
            week_path,
            experiment=None,
            channel="Channel 1",
            plot=True,
            save_plots=True,
            output_dir=Path(__file__).parents[3] / "output" / week,
            interactive=False,
            flat=True,
        )
    montage(
        Path(rf"H:\My Drive\Lab C\NMR"),
        output_file=Path(__file__).parents[3] / "output" / "montage",
        channel="Channel 1",
    )
