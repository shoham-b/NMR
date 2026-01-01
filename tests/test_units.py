import h5py
import numpy as np
from pathlib import Path
import os
from nmr_analysis.io.loader import KeysightLoader


def create_h5_with_units(filename):
    path = Path(filename)
    if path.exists():
        os.remove(path)

    with h5py.File(path, "w") as f:
        # Standard Keysight structure
        root = f.create_group("__BV_Dataset__Data__")

        # Data group (assume Channel 1)
        g1 = root.create_group("data_chan1_capture1")
        dset1 = g1.create_dataset("data_chan1_capture1", data=np.ones(100))
        dset1.attrs["XIncrement"] = 0.1
        dset1.attrs["XOrigin"] = 0.0

        # Unit structure
        # /__BV_Dataset__Data__/xdata_chan/BVAxisUnitLabel
        xdata_group = root.create_group("xdata_chan")
        # Creating BVAxisUnitLabel as a dataset containing the string "s" (seconds)
        # Note: h5py strings are tricky, using fixed width or variable.
        # Let's assume standard string.
        xdata_group.create_dataset(
            "BVAxisUnitLabel", data=np.array(["s"], dtype="S")
        )  # ASCII 's'

    return path


def test_unit_extraction():
    base_dir = Path("tests/temp_units")
    base_dir.mkdir(exist_ok=True, parents=True)
    filename = base_dir / "unit_test.h5"

    create_h5_with_units(filename)

    loader = KeysightLoader(channel="Channel 1")
    try:
        data = loader.load(filename)
        print("Loaded data.")
        if "time_unit" in data.metadata:
            print(f"Time unit found: {data.metadata['time_unit']}")
            if data.metadata["time_unit"] == "s":
                print("SUCCESS: Unit matches 's'")
            else:
                print(
                    f"FAILURE: Unit mismatch. Expected 's', got '{data.metadata['time_unit']}'"
                )
        else:
            print("FAILURE: 'time_unit' not found in metadata.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_unit_extraction()
