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
    extract_peak_by_index,
)
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
from nmr_analysis.io.loader import KeysightLoader
from nmr_analysis.visualization.interactive import generate_dashboard, AnalysisContext
from scipy.ndimage import gaussian_filter1d

ANALYSIS_SMOOTHING = 1.6

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
        subdirs = {
            "t1": ExperimentType.T1,
            "t2": ExperimentType.T2,
            "t2~": ExperimentType.T2_STAR,
            "t2_star": ExperimentType.T2_STAR,
            "t2multiple": ExperimentType.T2_COMBINED,
        }

        found_any = False
        for name, exp_type in subdirs.items():
            dataset_path = path / name
            if dataset_path.exists() and dataset_path.is_dir():
                found_any = True
                console.rule(
                    f"[bold cyan]Batch Analysis: {name} ({exp_type.value})[/bold cyan]"
                )
                try:
                    # Create subdirectory for this experiment type if saving
                    save_path = None
                    if save_plots:
                        save_path = output_dir / name
                        save_path.mkdir(parents=True, exist_ok=True)

                    ctxs = _run_analysis(
                        dataset_path, exp_type, channel, plot, save_path=save_path
                    )
                    if ctxs:
                        collected_contexts.extend(ctxs)
                except Exception as e:
                    console.print(f"[red]Failed to analyze {name}: {e}[/red]")

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
    loader = KeysightLoader(channel=channel)
    results = []

    if experiment == ExperimentType.T2_STAR:
        # T2*: Analyze each file independently
        target_files = []
        if path.is_dir():
            target_files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))
            if not target_files:
                raise FileNotFoundError(f"No HDF5 files in {path}")
        else:
            target_files = [path]

        console.print(f"Found {len(target_files)} T2* files to analyze.")

        for target_file in target_files:
            try:
                console.print(f"Loading {target_file.name}...")
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
        # Assuming directory of files or single file?
        # "From the combined we get the whole fit from a single measurment that has multiple echos"
        # So likely one file in t2combined dir.
        target_file = path
        if path.is_dir():
            files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))
            if not files:
                raise FileNotFoundError(f"No HDF5 files in {path}")
            target_file = files[0]

        console.print(f"Loading {target_file}...")
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
        params, fit_curve, residuals, r2 = Fitter.fit_t2(peak_times, peak_amps)

        result = AnalysisResult(
            experiment_type=experiment,
            dataset_name="Spin Echo (Echo Train)",
            params=params,
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
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

        files = list(path.glob(("*.h5"))) + list(path.glob(("*.hdf5")))
        if not files:
            console.print("[red]No .h5 or .hdf5 files found in directory.[/red]")
            raise typer.Exit(1)

        console.print(f"Found {len(files)} files. Processing...")

        delays = []
        amplitudes = []
        raw_traces = []

        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))
            for f in files:
                try:
                    data = loader.load(f)
                    # Determine peak index based on experiment type
                    # User request: "take the data starting from the first peak"
                    peak_idx_to_use = 0

                    # Extract peak with smoothing and robust constraints
                    # T1/T2 usually need specific peak. If unsure, user might want to adjust index.
                    t, amp, idx, all_peaks = extract_peak_by_index(
                        data,
                        peak_index=peak_idx_to_use,
                        smoothing=ANALYSIS_SMOOTHING,
                        min_distance=10,
                        min_height=5,
                        min_time_sep=0.1,
                    )

                    # Slice data starting from the peak
                    data.signal = data.signal[idx:]
                    data.time = data.time[idx:]

                    # Update peak index relative to new slice (it's now 0)
                    all_peaks = [p - idx for p in all_peaks if p >= idx]

                    # Delay is strictly the peak time (which is now the first point?)
                    # Wait, if we slice time, t is still the absolute time of the peak
                    tau = t

                    # Extract sort key and label from filename
                    # User request: order like "0_01" or "0_{number}", label to be that number
                    # Default: use tau if no match
                    import re

                    match = re.search(r"(0_[\d\.]+)", f.stem)
                    if match:
                        # User request: "0_022 is actually a decimal 0.022"
                        label = match.group(1).replace("_", ".")
                        # Try to convert to float for sorting (0_01 -> 0.01)
                        try:
                            sort_val = float(label)
                        except ValueError:
                            sort_val = tau  # Fallback
                    else:
                        label = f.stem
                        sort_val = tau

                    # Store label in metadata for plotting
                    data.metadata["trace_label"] = label

                    delays.append(t)
                    amplitudes.append(amp)
                    # Add sort_val to tuple for sorting
                    raw_traces.append((data, t, amp, tau, all_peaks, sort_val))
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

        # Sort raw traces by extracted sort_val (from filename number)
        # raw_traces structure: (data, t, amp, tau, all_peaks, sort_val)
        raw_traces.sort(key=lambda x: x[5])

        console.print("Fitting data...")
        if experiment == ExperimentType.T1:
            params, fit_curve, residuals, r2 = Fitter.fit_t1(delays, amplitudes)
            dataset_name = "T1 Analysis"
        else:  # T2
            params, fit_curve, residuals, r2 = Fitter.fit_t2(delays, amplitudes)
            dataset_name = "T2 Analysis"

        result = AnalysisResult(
            experiment_type=experiment,
            dataset_name=dataset_name,
            params=params,
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
        )

        print_result(result)
        if plot:
            filepath_fit = None
            filepath_traces = None
            if save_path:
                # Name based on directory?
                dirname = path.name
                filepath_fit = save_path / f"{dirname}_{experiment.value}_fit.png"
                filepath_traces = save_path / f"{dirname}_{experiment.value}_traces.png"
                console.print(f"Saving fit plot to {filepath_fit}")
                console.print(f"Saving traces plot to {filepath_traces}")

            # 1. Stacked Traces Plot
            plot_stacked_traces(
                raw_traces,
                filepath=filepath_traces,
                smoothing=ANALYSIS_SMOOTHING,
            )

            # 2. Fit Summary Plot (Raw Overlaid | Log Data)
            plot_analysis_summary(
                delays,
                amplitudes,
                result,
                raw_traces,
                # Use "Delay (s)" since we parsed it
                "Delay (s)",
                "Amplitude",
                filepath=filepath_fit,
                smoothing=ANALYSIS_SMOOTHING,
            )

        # For T1/T2, constructing 'data' representing the XY for plot
        # passing delays as time, amplitudes as signal
        aggregated_data = NMRData(time=delays, signal=amplitudes)
        # For T1/T2, constructing 'data' representing the XY for plot
        # passing delays as time, amplitudes as signal
        aggregated_data = NMRData(time=delays, signal=amplitudes)
        return [
            AnalysisContext(data=aggregated_data, result=result, raw_traces=raw_traces)
        ]


def print_result(result: AnalysisResult):
    table = Table(title=f"Results: {result.dataset_name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in result.params.items():
        table.add_row(k, f"{v:.4e}")

    table.add_row("R-Squared", f"{result.r_squared:.4f}")
    console.print(table)


def plot_result(
    x, y, result: AnalysisResult, xlabel, ylabel, filepath: Optional[Path] = None
):
    # T2* specific: only shows the log graph as requested
    plt.figure(figsize=(8, 6))

    # Filter start index if available (e.g. for T2*)
    start_index = 0
    if result.metadata and "start_index" in result.metadata:
        start_index = result.metadata["start_index"]

    x_plot = x[start_index:]
    y_plot = y[start_index:]

    # Filter for y > 1 (log friendly)
    mask = y_plot > 1

    plt.plot(x_plot[mask], y_plot[mask], label="Data", color="blue")

    if result.fit_curve is not None:
        fit_curve_plot = result.fit_curve[start_index:]
        # fit_curve might be fully NaN if fit failed, or partially NaN before start_idx
        # We sliced it so it should be fine, but check for > 1 and finite
        fit_mask = (fit_curve_plot > 1) & np.isfinite(fit_curve_plot)
        plt.plot(x_plot[fit_mask], fit_curve_plot[fit_mask], label="Fit", color="red")

    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (Log)")
    plt.title(f"{result.dataset_name} (Log Scale)")
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)

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
    # Raw Echo Train
    ax1.plot(
        data.time,
        np.abs(data.signal),
        label="Raw Echo Train",
        color="skyblue",
        alpha=0.6,
    )

    # Peaks (Scatter)
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

    # Fit Curve (Linear)
    if "M0" in result.params and "T2" in result.params:
        M0 = result.params["M0"]
        T2 = result.params["T2"]
        offset = result.params.get("offset", 0.0)
        full_fit_curve = t2_decay_model(data.time, M0, T2, offset)
        label_fit = f"T2 Fit (T2={T2:.4e} {unit})"

        ax1.plot(
            data.time,
            full_fit_curve,
            label=label_fit,
            color="red",
            linestyle="-",
            zorder=6,
        )
    else:
        ax1.plot(peak_times, result.fit_curve, label="Fit", color="red", zorder=6)

    ax1.set_xlabel(f"Time ({unit})")
    ax1.set_ylabel("Signal Magnitude")
    ax1.set_title(f"{result.dataset_name} (Linear)")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="best")

    # --- Plot 2: Decay (Log) ---
    # Only Peaks and Fit
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

    # Fit Curve (Log) - Plot against peak times for cleaner look on log
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
    raw_traces: List[Tuple[NMRData, float, float, float, np.ndarray]],
    filepath: Optional[Path] = None,
    smoothing: float = 1.0,
):
    """
    Plot each raw trace in its own subplot, stacked vertically.
    """
    num_traces = len(raw_traces)
    if num_traces == 0:
        return

    # Dynamic height: e.g., 2 inches per trace, min 6, max 50?
    fig_height = max(6, num_traces * 3)
    fig, axes = plt.subplots(num_traces, 1, figsize=(12, fig_height), sharex=True)

    if num_traces == 1:
        axes = [axes]

    # Color Mapping
    cmap = cm.viridis
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    for i, (data, t_peak, amp, tau, all_peaks, *_) in enumerate(raw_traces):
        ax = axes[i]
        color = cmap(norm(i))
        signal = np.abs(data.signal)

        # Raw trace (faint)
        ax.plot(data.time, signal, color=color, alpha=0.3)

        # Smoothed trace (bold)
        smoothed = gaussian_filter1d(signal, sigma=smoothing)
        ax.plot(
            data.time, smoothed, color=color, alpha=0.8, linestyle="-", linewidth=1.5
        )

        # Highlight selected peak
        ax.scatter(
            [t_peak],
            [amp],
            color="black",
            marker="x",
            s=80,
            zorder=5,
            label="Selected Peak",
        )

        # Plot ALL found peaks
        if len(all_peaks) > 0:
            ax.scatter(
                data.time[all_peaks],
                smoothed[all_peaks],
                color="gray",
                marker=".",
                s=30,
                zorder=4,
                alpha=0.6,
                label="Other Peaks",
            )

        unit = data.metadata.get("time_unit", "s")
        ax.set_ylabel("Amplitude")

        # User request: "name on top of the graph should be that number"
        if "trace_label" in data.metadata:
            ax.set_title(data.metadata["trace_label"])
        else:
            ax.set_title(f"Trace {i + 1}: $\\tau$={tau:.2e} {unit}")

        ax.grid(True, alpha=0.5)
        # Only legend on first or all? Maybe just markers match.
        # ax.legend(loc="upper right", fontsize="small")

    axes[-1].set_xlabel(f"Time ({raw_traces[0][0].metadata.get('time_unit', 's')})")

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
    for i, (data, t_peak, amp, tau, all_peaks, *_) in enumerate(raw_traces):
        color = cmap(norm(i))
        signal = np.abs(data.signal)
        # Raw trace (faint)
        ax_traces.plot(data.time, signal, color=color, alpha=0.3)

        # Smoothed trace (bold)
        smoothed = gaussian_filter1d(signal, sigma=smoothing)
        ax_traces.plot(
            data.time, smoothed, color=color, alpha=0.8, linestyle="-", linewidth=1.5
        )

        # Highlight selected peak
        ax_traces.scatter(
            [t_peak],
            [amp],
            color="black",
            marker="x",
            s=80,
            zorder=5,
            label="Selected Peak" if i == 0 else None,
        )

        # Plot ALL found peaks
        if len(all_peaks) > 0:
            ax_traces.scatter(
                data.time[all_peaks],
                smoothed[all_peaks],
                color="gray",
                marker=".",
                s=30,
                zorder=4,
                alpha=0.6,
                label="Other Peaks" if i == 0 else None,
            )

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
    for week in ("2.2",):
        analyze(
            Path(rf"H:\My Drive\Lab C\NMR\week{week}"),
            experiment=None,
            channel="Channel 1",
            plot=True,
            save_plots=True,
            output_dir=Path(__file__).parents[3] / "output" / week,
            interactive=True,
        )
