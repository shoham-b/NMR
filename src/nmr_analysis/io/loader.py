from pathlib import Path

import h5py
import numpy as np

from nmr_analysis.core.types import NMRData


class KeysightLoader:
    """Loader for Keysight HDF5 files."""

    def __init__(self, channel: str = "Channel 1"):
        self.channel = channel

    def load(self, file_path: Path) -> NMRData:
        """
        Load data from a Keysight HDF5 file.

        Args:
            file_path: Path to the HDF5 file.

        Returns:
            NMRData object containing time and signal arrays.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with h5py.File(file_path, "r") as f:
                # Strict requirement: /__BV_Dataset__Data__/data_chan{n}_capture1/data_chan{n}_capture1

                # Parse channel number from self.channel (e.g. "Channel 1" -> "1")
                import re

                match = re.search(r"(\d+)", self.channel)
                if not match:
                    raise ValueError(
                        f"Could not parse channel number from '{self.channel}'"
                    )
                channel_num = match.group(1)

                # Construct expected path
                # Path: /__BV_Dataset__Data__/data_chan{n}_capture1/data_chan{n}_capture1

                # Note: h5py allows access via full path
                group_path = f"/__BV_Dataset__Data__/data_chan{channel_num}_capture1"
                dataset_name = f"data_chan{channel_num}_capture1"

                if group_path not in f:
                    raise ValueError(
                        f"Group not found: {group_path}. File keys: {list(f.keys())}"
                    )

                group = f[group_path]

                if dataset_name not in group:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in group '{group_path}'. keys: {list(group.keys())}"
                    )

                dataset = group[dataset_name]

                raw_data = dataset[:]
                attrs = dict(dataset.attrs)
                # Merge group attrs
                attrs.update(dict(group.attrs))

                # Metadata extraction
                # Try to extract time unit from specific Bruker/Keysight path
                unit_path = "/__BV_Dataset__Data__/xdata_chan/BVAxisUnitLabel"
                try:
                    if unit_path in f:
                        unit_data = f[unit_path][()]
                        # Handle numpy array or scalar
                        if hasattr(unit_data, "flatten"):
                            unit_data = unit_data.flatten()
                            if len(unit_data) > 0:
                                unit_data = unit_data[0]

                        # Handle bytes to string decoding
                        if isinstance(unit_data, (bytes, np.bytes_)):
                            unit_data = unit_data.decode("utf-8")

                        attrs["time_unit"] = str(unit_data)
                except Exception:
                    # Ignore errors in unit extraction
                    pass

                # Metadata extraction
                x_inc = attrs.get("XIncrement", 1.0)
                x_org = attrs.get("XOrigin", 0.0)

                points = len(raw_data)

                time = np.linspace(x_org, x_org + (points - 1) * x_inc, points)

                return NMRData(time=time, signal=raw_data, metadata=attrs)

        except Exception as e:
            raise RuntimeError(f"Failed to load Keysight file {file_path}: {e}")
