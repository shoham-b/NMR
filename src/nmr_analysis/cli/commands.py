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
from nmr_analysis.analysis.models import t2_decay_model
from nmr_analysis.analysis.processing import (
    extract_echo_train,
    preprocess_data,
)
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
from nmr_analysis.io.loader import get_loader
from nmr_analysis.visualization.interactive import generate_dashboard, AnalysisContext

ANALYSIS_SMOOTHING = 2.6

app = typer.Typer()
console = Console()


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
):
    """
    Run analysis on NMR data. Supports batch processing of subdirectories.
    """
    collected_contexts: List[AnalysisContext] = []

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
                exp_type = ALIAS_MAP[name_lower]
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
                            out_sub = output_dir / item.name / sub.name
                            out_sub.mkdir(parents=True, exist_ok=True)
                        tasks.append((sub, out_sub))
                else:
                    # Flat/Standard mode: Process the folder itself
                    out_std = None
                    if save_plots:
                        out_std = output_dir / item.name
                        out_std.mkdir(parents=True, exist_ok=True)
                    tasks.append((item, out_std))

                for target_path, target_out in tasks:
                    console.rule(
                        f"[bold cyan]Batch Analysis: {target_path.name} ({exp_type.value})[/bold cyan]"
                    )
                    try:
                        ctxs = _run_analysis(
                            target_path, exp_type, channel, plot, save_path=target_out
                        )
                        if ctxs:
                            collected_contexts.extend(ctxs)
                    except Exception as e:
                        console.print(
                            f"[red]Failed to analyze {target_path.name}: {e}[/red]"
                        )

        if found_any:
            console.print("[green]Batch analysis completed.[/green]")
            if interactive and collected_contexts:
                output_html = output_dir / "index.html"
                generate_dashboard(collected_contexts, output_html)
                console.print(
                    f"[green]Interactive report saved to {output_html}[/green]"
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


def _run_analysis(
    path: Path,
    experiment: ExperimentType,
    channel: str,
    plot: bool,
    save_path: Optional[Path] = None,
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
                        filepath = out_dir / f"{target_file.stem}_fit.png"
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

        console.print("Extracting Echo Train...")
        # Paramaters for peak finding might need tuning or exposing
        # Using defaults for now, with min_height=0.5 to filter noise
        # Paramaters for peak finding might need tuning or exposing
        # Using defaults for now, with min_height=0.5 to filter noise
        # Paramaters for peak finding might need tuning or exposing
        # Using defaults for now, with min_height=0.5 to filter noise
        # User requested smoothing for peak finding
        peak_times, peak_amps = extract_echo_train(data, smoothing=ANALYSIS_SMOOTHING)

        if len(peak_times) < 3:
            console.print(
                "[red]Not enough peaks found for T2 fit (need at least 3, so >2).[/red]"
            )
            console.print(
                "[red]Not enough peaks found for T2 fit (need at least 3, so >2).[/red]"
            )
            return []

        # Skip the first 2 peaks (start from 3rd peak onward)
        peak_times = peak_times[2:]
        peak_amps = peak_amps[2:]

        console.print(f"Using {len(peak_times)} peaks (skipped first 2). Fitting T2...")

        # Fit T2 to the peaks
        # Using 0 as initial time? Use relative time?
        # Standard T2 fit: M(t) = M0 exp(-t/T2)
        # Delays are peak_times

        # Re-use T2 fitting logic
        params, fit_curve, residuals, r2, param_errors = Fitter.fit_t2(
            peak_times, peak_amps
        )

        result = AnalysisResult(
            experiment_type=experiment,
            dataset_name="Spin Echo (Echo Train)",
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
                filepath = save_path / f"{target_file.stem}_combined_fit.png"
                console.print(f"Saving plot to {filepath}")

            plot_combined_t2(data, peak_times, peak_amps, result, filepath=filepath)

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
                    data = loader.load(f)

                    # Preprocess: Find peak, slice, and shift time to 0
                    # Returns processed_data, tau, amp, peak_info
                    data, original_tau, amp, peak_info = preprocess_data(
                        data,
                        smoothing=ANALYSIS_SMOOTHING,
                    )

                    tau = original_tau

                    # Extract sort key and label from filename
                    import re

                    match = re.search(r"(0_[\d\.]+)", f.stem)
                    if match:
                        label = match.group(1).replace("_", ".")
                        try:
                            sort_val = float(label)
                        except ValueError:
                            sort_val = tau
                    else:
                        label = f.stem
                        sort_val = tau

                    data.metadata["trace_label"] = label

                    delays.append(original_tau)
                    amplitudes.append(amp)

                    # Store peak_info in the tuple
                    # Tuple structure: (data, peak_time(0.0), amp, tau, peak_info, sort_val)
                    # Note: We pass 0.0 as peak time because data is shifted to start at P1 (0.0)
                    # But peak_info indices refer to ORIGINAL data usually?
                    # Wait, preprocess_data returns NEW data sliced.
                    # P3 index relative to new data?
                    # preprocess_data returns:
                    # tau = t_fit - t_start (Difference)
                    # new_time starts at 0.
                    # t_fit in new time is exactly `tau`.
                    # So P3 is at `tau` in the new time axis.
                    # P1 is at 0.0.
                    # We can visualize P3 at `tau`.

                    # However, raw_traces usually stores processed data.
                    # So we should plot markers relative to this processed data.
                    # P1 is at 0.
                    # P3 is at tau.
                    # P2? We don't have P2 relative time easily unless we calc it.
                    # peak_info has INDICES into ORIGINAL data.
                    # We need indices into NEW data or Times.

                    # Let's trust peak_info has indices.
                    # p1_idx is start.
                    # p2_idx is original index.
                    # p3_idx is original index.
                    # New index = Old - p1_idx.
                    # So P2_new_idx = p2_idx - p1_idx (if > p1_idx).
                    # P3_new_idx = p3_idx - p1_idx.

                    # We'll calculate relative times for plotting.

                    raw_traces.append((data, 0.0, amp, tau, peak_info, sort_val))
                except Exception as e:
                    console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")

                progress.advance(task)

        if not delays:
            console.print("[red]No valid data processed.[/red]")
            raise typer.Exit(1)

        delays = np.array(delays)
        amplitudes = np.array(amplitudes)
        sorted_indices = np.argsort(delays)
        delays = delays[sorted_indices]
        amplitudes = amplitudes[sorted_indices]

        # raw_traces: (data, t_peak_dummy, amp, tau, peak_info, sort_val)
        raw_traces.sort(key=lambda x: x[5])

        console.print("Fitting data...")
        if experiment == ExperimentType.T1:
            params, fit_curve, residuals, r2, param_errors = Fitter.fit_t1(
                delays, amplitudes
            )
            dataset_name = "T1 Analysis"
        else:  # T2
            params, fit_curve, residuals, r2, param_errors = Fitter.fit_t2(
                delays, amplitudes
            )
            dataset_name = "T2 Analysis"

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
                dirname = path.name
                filepath_fit = save_path / f"{dirname}_{experiment.value}_fit.png"
                filepath_traces = save_path / f"{dirname}_{experiment.value}_traces.png"
                console.print(f"Saving fit plot to {filepath_fit}")
                console.print(f"Saving traces plot to {filepath_traces}")

            plot_stacked_traces(
                raw_traces,
                filepath=filepath_traces,
                smoothing=ANALYSIS_SMOOTHING,
            )

            plot_analysis_summary(
                delays,
                amplitudes,
                result,
                raw_traces,
                "Delay (s)",
                "Amplitude",
                filepath=filepath_fit,
                smoothing=ANALYSIS_SMOOTHING,
            )

        aggregated_data = NMRData(time=delays, signal=amplitudes)
        return [
            AnalysisContext(data=aggregated_data, result=result, raw_traces=raw_traces)
        ]


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
            if k in ("T2", "T1"):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, f"{v:.4e}")
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
    else:
        plt.show()


def plot_combined_t2(
    data: NMRData,
    peak_times: np.ndarray,
    peak_amps: np.ndarray,
    result: AnalysisResult,
    filepath: Optional[Path] = None,
):
    """
    Plot Raw Data, Peaks, and Fit Curve on a split graph (Linear | Log).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    unit = data.metadata.get("time_unit", "s")

    # --- Plot 1: Full Data (Linear) ---
    ax1.plot(
        data.time,
        np.abs(data.signal),
        label="Raw Echo Train",
        color="skyblue",
        alpha=0.6,
    )

    num_peaks = len(peak_times)
    cmap = cm.viridis
    norm = plt.Normalize(0, num_peaks - 1 if num_peaks > 1 else 1)
    colors = [cmap(norm(i)) for i in range(num_peaks)]

    ax1.scatter(
        peak_times,
        peak_amps,
        c=colors,
        marker="x",
        s=80,
        linewidths=2,
        zorder=5,
        label="_nolegend_",
    )

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
        ax1.plot(peak_times, result.fit_curve, label="Fit", color="red", zorder=6)

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
    else:
        plt.show()


def plot_stacked_traces(
    raw_traces: List[Tuple[NMRData, float, float, float, dict, float]],
    filepath: Optional[Path] = None,
    smoothing: float = 1.0,
):
    """
    Plot raw traces (left column) and smoothed traces (right column), stacked vertically.
    """
    num_traces = len(raw_traces)
    if num_traces == 0:
        return

    fig_height = max(6, num_traces * 3)
    # Create 2 columns: Raw (left) and Smoothed (right)
    fig, axes = plt.subplots(num_traces, 2, figsize=(16, fig_height), sharex=True)

    # Handle single trace case
    if num_traces == 1:
        axes = axes.reshape(1, 2)

    cmap = cm.viridis
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    for i, (data, t_peak, amp, tau, peak_info, *_) in enumerate(raw_traces):
        ax_raw = axes[i, 0]  # Left column: Raw
        ax_smooth = axes[i, 1]  # Right column: Smoothed
        color = cmap(norm(i))
        signal = np.abs(data.signal)
        smoothed = gaussian_filter1d(signal, sigma=smoothing)

        # --- LEFT: Raw Data ---
        ax_raw.plot(data.time, signal, color=color, alpha=0.8, linewidth=1.2)
        ax_raw.set_ylabel("Amplitude")
        ax_raw.grid(True, alpha=0.3)

        if "trace_label" in data.metadata:
            ax_raw.set_title(f"Raw: {data.metadata['trace_label']}")
        else:
            unit = data.metadata.get("time_unit", "s")
            ax_raw.set_title(f"Raw Trace {i + 1}: τ={tau:.2e} {unit}")

        # --- RIGHT: Smoothed Data with Peak Markers ---
        ax_smooth.plot(data.time, smoothed, color=color, alpha=0.9, linewidth=1.5)

        # Helper to plot peak markers
        def mark_peak(ax, idx_offset, color_marker, marker, label):
            if idx_offset >= 0 and idx_offset < len(data.time):
                ax.scatter(
                    [data.time[idx_offset]],
                    [smoothed[idx_offset]],
                    color=color_marker,
                    marker=marker,
                    s=100,
                    zorder=6,
                    label=label,
                    edgecolors="black",
                )

        p1_idx_orig = peak_info.get("p1_idx", 0)

        # P1 (Start) - Cyan Circle
        mark_peak(ax_smooth, 0, "cyan", "o", "P1 (Start)")

        # P2 (Noise) - Red X
        p2_idx_orig = peak_info.get("p2_idx", -1)
        if p2_idx_orig != -1 and p2_idx_orig >= p1_idx_orig:
            mark_peak(ax_smooth, p2_idx_orig - p1_idx_orig, "red", "X", "P2 (Ignored)")

        # Fit Peak - Green Star
        fit_idx_orig = peak_info.get("fit_idx", peak_info.get("p3_idx", 0))
        if fit_idx_orig >= p1_idx_orig:
            mark_peak(ax_smooth, fit_idx_orig - p1_idx_orig, "lime", "*", "Fit Peak")

        ax_smooth.set_ylabel("Amplitude")
        ax_smooth.grid(True, alpha=0.3)
        ax_smooth.legend(loc="best", fontsize=8)

        if "trace_label" in data.metadata:
            ax_smooth.set_title(
                f"Smoothed (σ={smoothing}): {data.metadata['trace_label']}"
            )
        else:
            unit = data.metadata.get("time_unit", "s")
            ax_smooth.set_title(f"Smoothed Trace {i + 1}: τ={tau:.2e} {unit}")

    # Set x-labels only on bottom row
    unit = raw_traces[0][0].metadata.get("time_unit", "s")
    axes[-1, 0].set_xlabel(f"Time ({unit})")
    axes[-1, 1].set_xlabel(f"Time ({unit})")

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
):
    """
    Plot Fit Result and Raw Traces in a split figure:
    1. Raw Traces (faint) + Smoothed Traces (bold) + Selected Peaks (Overlaid)
    2. Fit (Log)
    """
    fig, (ax_traces, ax_log) = plt.subplots(1, 2, figsize=(16, 6))

    # Color Mapping
    cmap = cm.viridis
    num_traces = len(raw_traces)
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    # --- Plot 1: Raw Traces (Time Domain) ---
    for i, (data, t_peak, amp, tau, peak_info, *_) in enumerate(raw_traces):
        color = cmap(norm(i))
        signal = np.abs(data.signal)
        # Raw trace (faint)
        ax_traces.plot(data.time, signal, color=color, alpha=0.3)

        # Smoothed trace (bold)
        smoothed = gaussian_filter1d(signal, sigma=smoothing)
        ax_traces.plot(
            data.time, smoothed, color=color, alpha=0.8, linestyle="-", linewidth=1.5
        )

        # Highlight 3 Peaks (P1, P2, P3)
        def mark_peak(idx_offset, color, marker, label):
            if idx_offset >= 0 and idx_offset < len(data.time):
                ax_traces.scatter(
                    [data.time[idx_offset]],
                    [smoothed[idx_offset]],
                    color=color,
                    marker=marker,
                    s=80,
                    zorder=5,
                    label=label if i == 0 else None,
                    edgecolors="black",
                )

        p1_idx_orig = peak_info.get("p1_idx", 0)

        # P1 (Start)
        mark_peak(0, "cyan", "o", "P1 (Start)")

        # P2 (Noise)
        p2_idx_orig = peak_info.get("p2_idx", -1)
        if p2_idx_orig != -1 and p2_idx_orig >= p1_idx_orig:
            mark_peak(p2_idx_orig - p1_idx_orig, "red", "X", "P2 (Ignored)")

        # P3 (Fit) or P2 (Fit) - Green Star
        # We use 'fit_idx' from peak_info which tells us WHICH peak was used.
        fit_idx_orig = peak_info.get("fit_idx", peak_info.get("p3_idx", 0))
        if fit_idx_orig >= p1_idx_orig:
            mark_peak(fit_idx_orig - p1_idx_orig, "lime", "*", "Fit Peak")

    ax_traces.set_xlabel(f"Time ({raw_traces[0][0].metadata.get('time_unit', 's')})")
    ax_traces.set_ylabel("Signal Amplitude")
    ax_traces.set_title("Raw Traces & Selected Peaks")
    ax_traces.legend(loc="upper right")
    ax_traces.grid(True, alpha=0.5)

    # --- Plot 2: Fit (Log) ---
    # Plot data points
    ax_log.scatter(x, y, c="blue", label="Data Points", zorder=3)

    # Fit Curve
    if result.fit_curve is not None:
        sorted_pairs = sorted(zip(x, result.fit_curve))
        sx, sy = zip(*sorted_pairs)
        ax_log.plot(sx, sy, label="Fit", color="red", linestyle="--", zorder=6)

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
    else:
        text_str = ""

    if text_str:
        ax_log.text(
            0.95,
            0.95,
            text_str,
            transform=ax_log.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel(f"{ylabel} (Log)")
    ax_log.set_title(f"{result.dataset_name} (Log Scale)")
    ax_log.set_yscale("log")
    ax_log.set_ylim(bottom=1)
    ax_log.grid(True, which="both", alpha=0.5)
    ax_log.legend(loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    for week in ("3.2",):
        analyze(
            Path(rf"H:\My Drive\Lab C\NMR\week{week}"),
            experiment=None,
            channel="Channel 1",
            plot=True,
            save_plots=True,
            output_dir=Path(__file__).parents[3] / "output" / week,
            interactive=False,
        )
