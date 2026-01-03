import shutil
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
import numpy as np
import h5py

runner = CliRunner()


def verify_water_workflow():
    # Setup mock data
    root = Path("verify_data_water")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    # Structure: root/Water
    water_dir = root / "Water"
    water_dir.mkdir()

    # 1. T2 folder (Standard T2)
    t2_dir = water_dir / "t2"
    t2_dir.mkdir()

    # Create valid T2 data (CPMG)
    taus_t2 = [0.001, 0.002, 0.005, 0.010, 0.020]
    for tau in taus_t2:
        fname = t2_dir / f"0_{str(tau).replace('.', '_')}.h5"
        with h5py.File(fname, "w") as f:
            group_path = "__BV_Dataset__Data__/data_chan2_capture1"
            dataset_name = "data_chan2_capture1"
            grp = f.create_group(group_path)
            time = np.linspace(0, 0.1, 100)
            # T2 = 0.05s
            signal = 1000 * np.exp(-time / 0.05) + np.random.normal(0, 1, 100)
            sig_complex = signal + 1j * np.random.normal(0, 1, 100)
            dset = grp.create_dataset(dataset_name, data=sig_complex)
            dset.attrs["XOrigin"] = 0.0
            dset.attrs["XIncrement"] = time[1] - time[0]

    # 2. t2_multiple folder (Diffusion)
    diff_dir = water_dir / "t2_multiple"
    diff_dir.mkdir()

    # Create valid Diffusion data (Echo Trains)
    taus_diff = [0.0001, 0.0002, 0.0005, 0.001]
    for tau in taus_diff:
        fname = diff_dir / f"0_{str(tau).replace('.', '_')}.h5"
        with h5py.File(fname, "w") as f:
            group_path = "__BV_Dataset__Data__/data_chan2_capture1"
            dataset_name = "data_chan2_capture1"
            grp = f.create_group(group_path)
            time = np.linspace(0, 0.2, 2000)
            envelope = 1000 * np.exp(-time / 0.5)
            decay_factor = np.exp(-10 * tau)
            # Make sure we have peaks
            carrier = np.abs(np.cos(2 * np.pi * 50 * time))
            signal = envelope * carrier * decay_factor + np.random.normal(0, 10, 2000)
            sig_complex = signal + 1j * np.random.normal(0, 10, 2000)
            dset = grp.create_dataset(dataset_name, data=sig_complex)
            dset.attrs["XOrigin"] = 0.0
            dset.attrs["XIncrement"] = time[1] - time[0]

    output_dir = Path("verify_output_water")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print("--- Running verify_water_workflow ---")
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

    print(f"Exit Code: {result.exit_code}")
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)

    # Check assertions
    # 1. Output directory should exist
    if not output_dir.exists():
        print("FAILURE: Output dir not created.")
        return

    files = [p.name for p in output_dir.iterdir()]
    print("Output files:", files)

    # 2. Check for T2 fit plot
    # Logic: prefix="Water", dirname="t2", exp="t2" -> Water_t2_t2_fit.png
    t2_plot = "Water_t2_t2_fit.png"
    if t2_plot in files:
        print(f"SUCCESS: {t2_plot} found.")
    else:
        print(f"FAILURE: {t2_plot} missing.")

    # 3. Check for Diffusion fit plot
    # Logic: prefix="Water", exp="diffusion" -> Water_diffusion_fit.png
    diff_plot = "Water_diffusion_fit.png"
    if diff_plot in files:
        print(f"SUCCESS: {diff_plot} found.")
    else:
        print(f"FAILURE: {diff_plot} missing.")

    # 4. Check stdout for "Captured T2"
    if "Captured T2 for Diffusion constraint" in result.stdout:
        print("SUCCESS: T2 constraint capture message found in stdout.")
    else:
        print("FAILURE: T2 constraint capture message MISSING.")


if __name__ == "__main__":
    verify_water_workflow()
