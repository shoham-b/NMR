import shutil
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
import numpy as np
import h5py

runner = CliRunner()


def reproduce():
    # Setup mock data
    root = Path("reproduce_data")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    # Structure: Water/t2 (Simulating the user's case where t2 implies diffusion for water)
    water_dir = root / "Water"
    water_dir.mkdir()
    t2_dir = water_dir / "t2"
    t2_dir.mkdir()

    # Create dummy HDF5 files for Diffusion (Echo Train signals)
    taus = [0.0001, 0.0002, 0.0005, 0.001]
    for i, tau in enumerate(taus):
        fname = t2_dir / f"0_{str(tau).replace('.', '_')}.h5"
        with h5py.File(fname, "w") as f:
            group_path = "__BV_Dataset__Data__/data_chan2_capture1"
            dataset_name = "data_chan2_capture1"
            grp = f.create_group(group_path)

            time = np.linspace(0, 0.2, 2000)
            envelope = 1000 * np.exp(-time / 0.5)
            # Vary amplitude with tau to simulate diffusion decay?
            # R2_obs = R2 + D * (gamma*G*tau)^2 / 12 ?
            # Just make amplitude decay with tau
            decay_factor = np.exp(-10 * tau)

            carrier = np.abs(np.cos(2 * np.pi * 50 * time))
            signal = envelope * carrier * decay_factor + np.random.normal(0, 10, 2000)
            sig_complex = signal + 1j * np.random.normal(0, 10, 2000)

            dset = grp.create_dataset(dataset_name, data=sig_complex)
            dset.attrs["XOrigin"] = 0.0
            dset.attrs["XIncrement"] = time[1] - time[0]

    output_dir = Path("reproduce_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    # output_dir.mkdir() # Should be created by command

    print("Running analyze command...")
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
        if result.exception:
            print("EXCEPTION:", result.exception)

    # Check for plot
    expected_plot = output_dir / "Water_diffusion_fit.png"
    if expected_plot.exists():
        print(f"SUCCESS: {expected_plot} exists.")
    else:
        print(f"FAILURE: {expected_plot} NOT found.")
        if output_dir.exists():
            print("Output dir contents:", list(output_dir.iterdir()))
        else:
            print("Output dir does not exist.")


if __name__ == "__main__":
    reproduce()
