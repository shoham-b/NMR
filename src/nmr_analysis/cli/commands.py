from pathlib import Path
from typing import List, Optional, Tuple
import typer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from nmr_analysis.io.loader import KeysightLoader
from nmr_analysis.analysis.models import t2_decay_model
from nmr_analysis.analysis.processing import (
    extract_echo_train,
    extract_peak_by_index,
)
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
from nmr_analysis.visualization.interactive import generate_dashboard, AnalysisContext

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
                    ctx = _run_analysis(dataset_path, exp_type, channel, plot)
                    if ctx:
                        collected_contexts.append(ctx)
                except Exception as e:
                    console.print(f"[red]Failed to analyze {name}: {e}[/red]")

        if found_any:
            console.print("[green]Batch analysis completed.[/green]")
            if interactive and collected_contexts:
                output_html = path / "index.html"
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

    ctx = _run_analysis(path, experiment, channel, plot)
    if interactive and ctx:
        output_html = (path if path.is_dir() else path.parent) / "index.html"
        generate_dashboard([ctx], output_html)
        console.print(f"[green]Interactive report saved to {output_html}[/green]")


def _run_analysis(
    path: Path, experiment: ExperimentType, channel: str, plot: bool
) -> Optional[AnalysisContext]:
    loader = KeysightLoader(channel=channel)

    if experiment == ExperimentType.T2_STAR:
        # Expecting single file (can be directory if new t~ structure implies series, but usually FID is one)
        # Assuming single file for now or finding first file in dir
        target_file = path
        if path.is_dir():
            files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))
            if not files:
                raise FileNotFoundError(f"No HDF5 files in {path}")
            target_file = files[0]  # Pick first one or loop?
            # If t~ directory has multiple, user might mean batch of t2*?
            # Let's stick to single file behavior or loop if directory for robustness?
            # Existing plan: T2* single file.
            # If path is t~ directory, picking first file is safer for now.
            if len(files) > 1:
                console.print(
                    f"[yellow]Warning: Multiple files in {path}, analyzing {target_file.name}[/yellow]"
                )

        console.print(f"Loading {target_file}...")
        data = loader.load(target_file)
        console.print("Fitting T2* decay...")
        result = Fitter.fit_t2_star(data)
        print_result(result)
        if plot:
            plot_result(
                data.time, np.abs(data.signal), result, "Time (s)", "Signal (Magnitude)"
            )
        return AnalysisContext(data=data, result=result)

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
        peak_times, peak_amps = extract_echo_train(data, min_height=0.5)

        if len(peak_times) < 3:
            console.print(
                "[red]Not enough peaks found for T2 fit (need at least 3, so >2).[/red]"
            )
            return None

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
            dataset_name="T2 Combined (Echo Train)",
            params=params,
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
        )
        print_result(result)
        if plot:
            # We want: Raw Data + Peaks + Fit Curve on ONE graph
            plot_combined_t2(data, peak_times, peak_amps, result)

        return AnalysisContext(
            data=data, result=result, peak_times=peak_times, peak_amps=peak_amps
        )

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
                    # Extract from 3rd peak (index 2)
                    t, amp, idx = extract_peak_by_index(data, peak_index=2)

                    delays.append(t)
                    amplitudes.append(amp)
                    raw_traces.append((data, t, amp))
                except Exception as e:
                    console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")

                progress.advance(task)

        delays = np.array(delays)
        amplitudes = np.array(amplitudes)
        sorted_indices = np.argsort(delays)
        delays = delays[sorted_indices]
        amplitudes = amplitudes[sorted_indices]

        # Sort raw traces
        raw_traces.sort(key=lambda x: x[1])

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
            plot_analysis_summary(
                delays, amplitudes, result, raw_traces, "Delay (s)", "Amplitude"
            )

        # For T1/T2, constructing 'data' representing the XY for plot
        # passing delays as time, amplitudes as signal
        aggregated_data = NMRData(time=delays, signal=amplitudes)
        return AnalysisContext(
            data=aggregated_data, result=result, raw_traces=raw_traces
        )


def print_result(result: AnalysisResult):
    table = Table(title=f"Results: {result.dataset_name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in result.params.items():
        table.add_row(k, f"{v:.4e}")

    table.add_row("R-Squared", f"{result.r_squared:.4f}")
    console.print(table)


def plot_result(x, y, result: AnalysisResult, xlabel, ylabel, filepath: str = None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data", color="blue")
    plt.plot(x, result.fit_curve, label="Fit", color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{result.dataset_name} Fit")
    plt.legend()
    plt.grid(True)
    if filepath:
        plt.savefig(filepath)
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()


def plot_combined_t2(
    data: NMRData, peak_times: np.ndarray, peak_amps: np.ndarray, result: AnalysisResult
):
    """
    Plot Raw Data, Peaks, and Fit Curve on a single graph.
    Matches style of plot_analysis_summary.
    """
    plt.figure(figsize=(12, 8))

    # 1. Raw Data (Background)
    plt.plot(
        data.time,
        np.abs(data.signal),
        label="Raw Echo Train",
        color="lightgray",
        alpha=0.6,
        linewidth=1,
    )

    # 2. Peaks (Scatter with Gradient)
    # Use rank/index for color mapping
    num_peaks = len(peak_times)
    cmap = cm.viridis
    norm = plt.Normalize(0, num_peaks - 1 if num_peaks > 1 else 1)

    # Create colors for each peak
    colors = [cmap(norm(i)) for i in range(num_peaks)]

    plt.scatter(
        peak_times,
        peak_amps,
        c=colors,
        marker="x",
        s=100,
        linewidths=2,
        zorder=5,
        label="_nolegend_",  # Legend handled via dummy or assume 'Raw Echo Train' covers it?
        # Better to add dummy for "Extracted Peaks" styling
    )

    # Dummy marker for legend
    plt.scatter([], [], color="black", marker="x", label="Extracted Peaks")

    # 3. Fit Curve
    if "M0" in result.params and "T2" in result.params:
        M0 = result.params["M0"]
        T2 = result.params["T2"]
        offset = result.params.get("offset", 0.0)

        full_fit_curve = t2_decay_model(data.time, M0, T2, offset)

        plt.plot(
            data.time,
            full_fit_curve,
            label=f"T2 Fit (T2={T2 * 1e6:.2f} \u03bcs)",
            color="red",
            linewidth=2,
            linestyle="-",
            zorder=6,
        )
    else:
        # Fallback
        plt.plot(peak_times, result.fit_curve, label="Fit", color="red", zorder=6)

    plt.xlabel("Time (s)")
    plt.ylabel("Signal Magnitude")
    plt.title("T2 Combined Analysis: Raw Data, Peaks, and Fit")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.5)

    plt.ylim(auto=True)

    # Add colorbar for delay context
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, label="Peak Sequence (Delay)")

    # Custom ticks for colorbar (Delay values)
    if num_peaks > 1:
        tick_indices = np.linspace(0, num_peaks - 1, num=min(6, num_peaks), dtype=int)
        tick_indices = np.unique(tick_indices)
        tick_labels = [f"{peak_times[i]:.2e}" for i in tick_indices]
        cbar.set_ticks(tick_indices)
        cbar.set_ticklabels(tick_labels)

    plt.tight_layout()
    plt.show()


def plot_analysis_summary(
    x,
    y,
    result: AnalysisResult,
    raw_traces: List[Tuple[NMRData, float, float]],
    xlabel,
    ylabel,
):
    """
    Plot Fit Result and Raw Traces in a single figure.
    """
    plt.figure(figsize=(12, 8))

    # 1. Plot all Raw Traces (faint)
    cmap = cm.viridis
    # Use RANK (index) for color mapping
    num_traces = len(raw_traces)
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    # Iterate raw_traces (which were sorted by time)
    for i, (data, t, amp) in enumerate(raw_traces):
        color = cmap(norm(i))
        plt.plot(data.time, np.abs(data.signal), color=color, alpha=0.3, linewidth=1)

        # 2. Highlight specific peak used (Scatter)
        plt.scatter(
            [t],
            [amp],
            color=color,
            marker="x",
            s=100,
            linewidths=2,
            zorder=5,
            label="_nolegend_",
        )

    # 3. Plot the Fit Curve
    # Reconstruct smooth curve or just plot the fit points?
    # Ideally we plot the fit curve over the delay times.
    sorted_pairs = sorted(zip(x, result.fit_curve))
    sx, sy = zip(*sorted_pairs)
    plt.plot(
        sx,
        sy,
        label="T2 Fit",
        color="red",
        linewidth=2,
        linestyle="-",
        zorder=6,
    )

    # Add dummy marker for legend to explain the 'x'
    plt.scatter([], [], color="black", marker="x", label="Peaks (Data)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{result.dataset_name} - Single Plot Analysis")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.5)

    plt.ylim(auto=True)

    # Add colorbar for delay context
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, label="Peak Delay (s)")

    # Set ticks to show actual values based on index
    if num_traces > 1:
        # Select indices for ticks
        tick_indices = np.linspace(0, num_traces - 1, num=min(6, num_traces), dtype=int)
        tick_indices = np.unique(tick_indices)

        # Labels are the delays at those indices
        tick_labels = [f"{raw_traces[i][1]:.2e}" for i in tick_indices]

        cbar.set_ticks(tick_indices)
        cbar.set_ticklabels(tick_labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze(
        Path(r"H:\My Drive\Lab C\NMR"), experiment=None, channel="Channel 2", plot=True
    )
