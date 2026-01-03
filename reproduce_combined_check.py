import shutil
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
import numpy as np
import h5py

runner = CliRunner()


def reproduce_combined_case():
    # Setup mock data for T2 Combined nested in Water
    root = Path("reproduce_data_combined")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    # Structure: root/Water/t2_multiple
    water_dir = root / "Water"
    water_dir.mkdir()
    combined_dir = water_dir / "t2_multiple"
    combined_dir.mkdir()

    # Create dummy HDF5 files for Diffusion (Echo Train)
    taus = [0.0001, 0.0002, 0.0005, 0.001]
    for i, tau in enumerate(taus):
        fname = combined_dir / f"0_{str(tau).replace('.', '_')}.h5"
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

    output_dir = Path("reproduce_output_combined")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print("--- Scene 4: Analyze Water/t2_multiple (Batch default) ---")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(root),  # Running on root, containing Water/t2_multiple
            "--flat",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )
    print("Scene 4 Exit Code:", result.exit_code)

    # Expect: Water_t2_multiple_diffusion_fit.png (assuming prefix is Water in flat mode?)
    # Logic: prefix = "Water" (item.name) line 260
    # fname = f"{prefix}_diffusion_fit.png" line 512 => Water_diffusion_fit.png

    expected_plot = output_dir / "Water_diffusion_fit.png"

    if expected_plot.exists():
        print(
            f"SUCCESS: {expected_plot} exists. (Diffusion triggered for t2_multiple in Water)"
        )
    else:
        print(f"FAILURE: {expected_plot} NOT found.")
        print("Contents:", [p.name for p in output_dir.iterdir()])


if __name__ == "__main__":
    reproduce_combined_case()
