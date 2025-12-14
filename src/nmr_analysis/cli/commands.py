import typer
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import numpy as np
import matplotlib.pyplot as plt

from nmr_analysis.io.loader import KeysightLoader
from nmr_analysis.analysis.processing import (
    extract_peak_amplitude,
    parse_time_from_filename,
)
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.core.types import ExperimentType, AnalysisResult

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
        ..., help="Path to input file (T2*) or directory (T1/T2)"
    ),
    experiment: ExperimentType = typer.Option(
        ..., "-t", "--type", help="Type of experiment: t1, t2, t2_star"
    ),
    channel: str = typer.Option("Channel 1", help="Scope channel name"),
    plot: bool = typer.Option(True, help="Show plot of the fit"),
):
    """
    Run analysis on NMR data.
    """
    loader = KeysightLoader(channel=channel)

    if experiment == ExperimentType.T2_STAR:
        # Expecting single file
        if path.is_dir():
            console.print(
                "[red]Error: T2* analysis expects a single file, got directory.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Loading {path}...")
        try:
            data = loader.load(path)
            console.print("Fitting T2* decay...")
            result = Fitter.fit_t2_star(data)

            print_result(result)

            if plot:
                plot_result(
                    data.time,
                    np.abs(data.signal),
                    result,
                    "Time (s)",
                    "Signal (Magnitude)",
                )

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

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

        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))
            for f in files:
                try:
                    data = loader.load(f)
                    amp = extract_peak_amplitude(data, method="max_abs")

                    # Try to get time
                    t = parse_time_from_filename(f.name)
                    # If 0.0, maybe warn or try metadata (omitted metadata check for now as it's not implemented fully)

                    delays.append(t)
                    amplitudes.append(amp)
                except Exception as e:
                    console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")

                progress.advance(task)

        # Sort by delay
        delays = np.array(delays)
        amplitudes = np.array(amplitudes)
        sorted_indices = np.argsort(delays)
        delays = delays[sorted_indices]
        amplitudes = amplitudes[sorted_indices]

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
            plot_result(delays, amplitudes, result, "Delay (s)", "Amplitude")


def print_result(result: AnalysisResult):
    table = Table(title=f"Results: {result.dataset_name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in result.params.items():
        table.add_row(k, f"{v:.4e}")

    table.add_row("R-Squared", f"{result.r_squared:.4f}")
    console.print(table)


def plot_result(x, y, result: AnalysisResult, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data", color="blue")
    plt.plot(x, result.fit_curve, label="Fit", color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{result.dataset_name} Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    app()
