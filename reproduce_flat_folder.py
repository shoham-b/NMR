import shutil
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
import numpy as np
import h5py

runner = CliRunner()


def reproduce_flat_folder():
    # Setup mock data
    root = Path("reproduce_data_flat")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    # Structure: Water/*.h5
    water_dir = root / "Water"
    water_dir.mkdir()

    # Create dummy HDF5 files for Diffusion
    taus = [0.0001, 0.0002, 0.0005, 0.001]
    for i, tau in enumerate(taus):
        fname = water_dir / f"0_{str(tau).replace('.', '_')}.h5"
        with h5py.File(fname, "w") as f:
            group_path = "__BV_Dataset__Data__/data_chan2_capture1"
            dataset_name = "data_chan2_capture1"
            grp = f.create_group(group_path)
            time = np.linspace(0, 0.2, 2000)
            envelope = 1000 * np.exp(-time / 0.5)
            decay_factor = np.exp(-10 * tau)
            carrier = np.abs(np.cos(2 * np.pi * 50 * time))
            signal = envelope * carrier * decay_factor + np.random.normal(0, 10, 2000)
            sig_complex = signal + 1j * np.random.normal(0, 10, 2000)
            dset = grp.create_dataset(dataset_name, data=sig_complex)
            dset.attrs["XOrigin"] = 0.0
            dset.attrs["XIncrement"] = time[1] - time[0]

    output_dir = Path("reproduce_output_flat")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print("Running analyze command on Water folder (direct)...")
    # If user runs 'analyze data/Water', it should work as Single Analysis?
    # Or 'analyze data/'?

    # Scene 1: analyze data/ (root)
    print("--- Scene 1: Analyze root ---")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(root),
            "--flat",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )
    print("Scene 1 Exit Code:", result.exit_code)

    # Expect: Water is skipped unless it matches ALIAS_MAP?
    # Water is NOT in ALIAS_MAP.
    # It has no subdirs.
    # So it should be skipped in batch mode.
    # Result: No files.

    if not any(output_dir.iterdir()):
        print("Scene 1 Failure: No output files (Expected if Water is ignored).")

    # Scene 2: analyze data/Water explicitly
    # Must specify type?
    # Logic line 306: if experiment is None: Exit(1).

    print("--- Scene 2: Analyze Water explicit (no type) ---")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(water_dir),
            "--flat",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )
    print("Scene 2 Exit Code:", result.exit_code)  # Should be 1

    # Scene 3: Analyze Water explicit WITH --type diffusion
    print("--- Scene 3: Analyze Water explicit WITH --type diffusion ---")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(water_dir),
            "--type",
            "diffusion",
            "--flat",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )
    print("Scene 3 Exit Code:", result.exit_code)
    expected_plot = (
        output_dir / "diffusion_fit.png"
    )  # Prefix empty for single analysis default?
    # Or prefix = parent.name if flat?
    # verify logic in commands.py line 329: default prefix=""
    # single analysis call line 318: prefix not passed!

    if expected_plot.exists():
        print(f"SUCCESS: {expected_plot} exists.")
    else:
        print(f"FAILURE: {expected_plot} NOT found.")
        print("Contents:", [p.name for p in output_dir.iterdir()])


if __name__ == "__main__":
    reproduce_flat_folder()
