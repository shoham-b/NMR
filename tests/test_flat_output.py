import pytest
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
import numpy as np
import h5py

runner = CliRunner()


@pytest.fixture
def flat_test_data(tmp_path):
    """
    Creates a mock directory structure:
    root/
      Sample1/
        Diffusion/
          0_0001.h5
          0_0002.h5
          0_0005.h5
      Sample2/
        Diffusion/
           0_0001.h5
           0_0002.h5
           0_0005.h5
    """
    root = tmp_path / "mock_data"
    root.mkdir()

    for sample in ["Sample1", "Sample2"]:
        s_dir = root / sample
        s_dir.mkdir()
        diff_dir = s_dir / "Diffusion"
        diff_dir.mkdir()

        # Create dummy HDF5 files for Diffusion
        taus = [0.0001, 0.0002, 0.0005]
        for tau in taus:
            fname = diff_dir / f"0_{str(tau).replace('.', '_')}.h5"
            with h5py.File(fname, "w") as f:
                # Structure for Keysight loader (Channel 2 default)
                # Path: /__BV_Dataset__Data__/data_chan2_capture1/data_chan2_capture1

                group_path = "__BV_Dataset__Data__/data_chan2_capture1"
                dataset_name = "data_chan2_capture1"

                grp = f.create_group(group_path)

                # Mock signal: CPMG Echo Train
                # T2 = 0.5s decay of envelope
                # Echoes every 0.01s (100Hz effective echo rate)
                time = np.linspace(0, 0.2, 2000)
                envelope = 1000 * np.exp(-time / 0.5)
                # Carrier: peaks at 0, 0.01, 0.02, ...
                carrier = np.abs(np.cos(2 * np.pi * 50 * time))

                signal = envelope * carrier + np.random.normal(0, 10, 2000)
                sig_complex = signal + 1j * np.random.normal(0, 10, 2000)

                dset = grp.create_dataset(dataset_name, data=sig_complex)

                # Metadata
                dset.attrs["XOrigin"] = 0.0
                dset.attrs["XIncrement"] = time[1] - time[0]

    return root


def test_flat_output_structure(flat_test_data, tmp_path):
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(flat_test_data),
            "--flat",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    if result.exit_code != 0:
        with open("test_debug.log", "w", encoding="utf-8") as f:
            f.write(f"Exit Code: {result.exit_code}\n")
            f.write("STDOUT:\n")
            f.write(str(result.stdout))
            if result.exception:
                f.write(f"\nEXCEPTION: {result.exception}\n")
                import traceback

                traceback.print_tb(result.exc_info[2], file=f)

    assert result.exit_code == 0

    try:
        # Check that output directory exists
        assert output_dir.exists()

        # Check that NO subdirectories exist in output_dir
        subdirs = [p for p in output_dir.iterdir() if p.is_dir()]
        assert len(subdirs) == 0, f"Found subdirectories: {subdirs}"

        # Check for summary.csv
        summary_file = output_dir / "summary.csv"
        assert summary_file.exists()

        # Check contents of summary.csv
        import pandas as pd

        df = pd.read_csv(summary_file)
        # We expect 2 samples * 1 analysis each = 2 rows
        assert len(df) >= 2
        assert "Sample" in df.columns
        assert set(df["Sample"]).issuperset({"Sample1", "Sample2"})

        # Check for plots with prefixed names
        # Sample1_diffusion_fit.png
        # Sample2_diffusion_fit.png

        files = [f.name for f in output_dir.iterdir()]
        assert "Sample1_diffusion_fit.png" in files
        assert "Sample2_diffusion_fit.png" in files

    except AssertionError as e:
        with open("test_assertion.log", "w", encoding="utf-8") as f:
            f.write(f"Assertion Failed: {e}\n")
            f.write("STDOUT:\n")
            f.write(str(result.stdout))
            f.write("\nOutput Directory Contents:\n")
            if output_dir.exists():
                for p in output_dir.rglob("*"):
                    f.write(f"{p}\n")
        raise e
