from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import typer
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.models import t2_decay_model
from nmr_analysis.analysis.processing import (
    extract_echo_train,
    preprocess_data,
    compute_spectrum,
)
from nmr_analysis.analysis.hybrid import analyze_spectral_series, HybridAnalysisResult
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
from nmr_analysis.io.loader import get_loader
from nmr_analysis.io.reporting import save_summary_csv
from nmr_analysis.visualization.interactive import generate_dashboard, AnalysisContext
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from scipy.ndimage import gaussian_filter1d

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

            if name_lower in ALIAS_MAP or name_lower.endswith("nol"):
                if name_lower.endswith("nol"):
                    exp_type = ExperimentType.SPECTRUM
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
                        fname = (
                            f"{prefix}_{target_file.stem}_fit.png"
                            if prefix
                            else f"{target_file.stem}_fit.png"
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

            hybrid_res = analyze_spectral_series(data_list, names)

            # Print Summary
            console.print(
                f"[bold]Hybrid Analysis Results: {hybrid_res.dataset_name}[/bold]"
            )
            table = Table(title="Spectral T2 Results")
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
                # Use dataset name (clean) for plotting
                plot_hybrid_result(
                    hybrid_res,
                    out_dir,
                    prefix=f"{prefix_str}hybrid_{hybrid_res.dataset_name}",
                )

            return []

        else:
            # Standard Single File Analysis
            # User Request: "T2* is in the time domain as always was"
            # So even if Type=SPECTRUM, for single file, do Time Domain T2* Fit but Show Spectrum.
            for target_file in target_files:
                try:
                    console.print(f"Loading {target_file.name}...")
                    loader = get_loader(target_file, channel=channel)
                    data = loader.load(target_file)

                    # 1. Compute Spectrum for visualization
                    console.print("Computing Spectrum...")
                    freqs, spect = compute_spectrum(data)

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
                        ax.plot(freqs, np.abs(spect), color="black")
                        ax.set_title(f"Spectrum: {target_file.name}")
                        ax.set_xlabel("Frequency (Hz)")
                        ax.set_ylabel("Magnitude")
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
                fname = (
                    f"{prefix}_{target_file.stem}_combined_fit.png"
                    if prefix
                    else f"{target_file.stem}_combined_fit.png"
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

                    # Preprocess: Find peak, slice, and shift time to 0
                    # Returns processed_data, tau, amp, peak_info
                    processed_data, original_tau, amp, peak_info = preprocess_data(
                        data_full,
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

                    data_full.metadata["trace_label"] = label
                    processed_data.metadata["trace_label"] = label

                    delays.append(original_tau)
                    amplitudes.append(amp)

                    # Store full raw data for visualization (User Request)
                    # We pass data_full (untouched) but we also need peak_info to know where peaks are relative to it.
                    # peak_info contains 'trim_start_idx' which connects processed to full.
                    # Note: We rely on trim_start_idx to adjust plotting markers.

                    raw_traces.append(
                        (processed_data, data_full, 0.0, amp, tau, peak_info, sort_val)
                    )
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

        # raw_traces: (processed, full, t_peak, amp, tau, peak_info, sort_val)
        # Sort by sort_val (index 6)
        raw_traces.sort(key=lambda x: x[6])

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
                p_str = f"{prefix}_" if prefix else ""
                filepath_fit = (
                    save_path / f"{p_str}{dirname}_{experiment.value}_fit.png"
                )
                filepath_traces = (
                    save_path / f"{p_str}{dirname}_{experiment.value}_traces.png"
                )
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


def plot_spectrum_fit(freqs, mag_data, result, filepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, mag_data, label="Data (Magnitude)", color="black", alpha=0.7)
    if len(result.fit_curve) > 0:
        ax.plot(
            freqs,
            result.fit_curve,
            label="Fit (Mag Lorentzian)",
            color="red",
            linestyle="--",
        )

    # Mark peaks
    if "peaks" in result.params:
        for p in result.params["peaks"]:
            f0 = p["f0"]
            ax.axvline(f0, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{result.dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plot_hybrid_result(result: HybridAnalysisResult, out_dir: Path, prefix: str = ""):
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
        ax_freq.plot(freqs, mag + i * offset_step_f, color=color, alpha=0.8)

    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude (Stacked)")
    ax_freq.set_title(f"Stacked Spectra")
    ax_freq.grid(True, alpha=0.3)

    fig_stack.suptitle(f"Hybrid Analysis Overview: {result.dataset_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname_stack = f"{prefix}_stacked_overview.png"
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
        if fit_res.get("T2", 0) > 0:
            y_fit = t2_decay_model(
                t_smooth, fit_res["M0"], fit_res["T2"], fit_res["offset"]
            )
            ax_lin.plot(t_smooth, y_fit, "r--", label="Fit", linewidth=2, zorder=2)

        ax_lin.set_xlabel("Delay $\\tau$ (s)")
        ax_lin.set_ylabel("Integrated Area (Frequency Domain)")
        ax_lin.set_title(f"Peak @ {f0:.1f} Hz (Linear)")
        ax_lin.grid(True, alpha=0.5)
        ax_lin.legend(loc="best")

        # Right: Log
        ax_log = axes[k, 1]
        valid_mask = areas > 0
        ax_log.scatter(
            taus[valid_mask],
            areas[valid_mask],
            label="Integrated Area",
            color="blue",
            s=50,
            zorder=3,
        )

        if fit_res.get("T2", 0) > 0:
            y_fit_log = t2_decay_model(
                t_smooth, fit_res["M0"], fit_res["T2"], fit_res["offset"]
            )
            # Filter non-positive for log plot
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

        ax_log.set_yscale("log")
        ax_log.set_xlabel("Delay $\\tau$ (s)")
        ax_log.set_ylabel("Integrated Area (Log)")
        ax_log.set_title(f"Peak @ {f0:.1f} Hz (Log)")
        ax_log.grid(True, which="both", alpha=0.5)

    fig_fit.suptitle(f"T2 Decay Analysis: {result.dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname_fit = f"{prefix}_t2_decay.png"
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
    else:
        plt.show()


def plot_stacked_traces(
    raw_traces: List[Tuple[NMRData, NMRData, float, float, float, dict, float]],
    filepath: Optional[Path] = None,
    smoothing: float = 1.0,
):
    """
    Plot processed traces (left) and full raw traces (right), stacked vertically.
    """
    num_traces = len(raw_traces)
    if num_traces == 0:
        return

    fig_height = max(6, num_traces * 3)
    # Create 2 columns: Processed (left) and Raw (right)
    # Different x-axes
    fig, axes = plt.subplots(num_traces, 2, figsize=(16, fig_height))

    # Handle single trace case
    if num_traces == 1:
        axes = axes.reshape(1, 2)

    cmap = cm.viridis
    norm = plt.Normalize(0, num_traces - 1 if num_traces > 1 else 1)

    for i, (processed_data, data_full, t_peak, amp, tau, peak_info, *_) in enumerate(
        raw_traces
    ):
        ax_proc = axes[i, 0]  # Left column: Processed
        ax_raw = axes[i, 1]  # Right column: Full Raw
        color = cmap(norm(i))

        proc_signal = np.abs(processed_data.signal)
        full_signal = np.abs(data_full.signal)

        # --- LEFT: Processed Data with Peaks ---
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

        # --- RIGHT: Full Raw Data ---
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

        trim_offset = peak_info.get("trim_start_idx", 0)

        # P1 (Start)
        p1_idx_rel = peak_info.get("p1_idx", 0)
        p1_idx = trim_offset + p1_idx_rel
        mark_peak(p1_idx, "cyan", "o", "P1 (Start)")

        # P2 (Noise)
        p2_idx_rel = peak_info.get("p2_idx", -1)
        if p2_idx_rel != -1:
            p2_idx = trim_offset + p2_idx_rel
            if p2_idx >= p1_idx:
                mark_peak(p2_idx, "red", "X", "P2 (Ignored)")

        # P3 (Fit) or P2 (Fit) - Green Star
        # We use 'fit_idx' from peak_info which tells us WHICH peak was used.
        fit_idx_rel = peak_info.get("fit_idx", peak_info.get("p3_idx", 0))
        fit_idx = trim_offset + fit_idx_rel

        if fit_idx >= p1_idx:
            mark_peak(fit_idx, "lime", "*", "Fit Peak")

    ax_traces.set_xlabel(f"Time ({raw_traces[0][0].metadata.get('time_unit', 's')})")
    ax_traces.set_ylabel("Signal Amplitude")
    ax_traces.set_title("Raw Traces & Selected Peaks")
    ax_traces.legend(loc="upper right")
    ax_traces.grid(True, alpha=0.5)

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
    for week in ("4.2",):
        analyze(
            Path(rf"H:\My Drive\Lab C\NMR\week{week}"),
            experiment=None,
            channel="Channel 1",
            plot=True,
            save_plots=True,
            output_dir=Path(__file__).parents[3] / "output" / week,
            interactive=False,
            flat=True,
        )
