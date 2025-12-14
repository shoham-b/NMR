import h5py
import numpy as np
from pathlib import Path
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

        try:
            with h5py.File(file_path, "r") as f:
                # Structure of Keysight HDF5 usually varies, but typically:
                # /Waveforms/Channel 1/Element 1/Raw Data
                # Or simply stored under a group.
                # This is a generic implementation that might need adjustment based on specific file structure.
                # Assuming 'Waveforms' group exists.

                # We need to explore the file to find the data if structure is unknown,
                # but based on prompt "specific channel", we assume a path.

                # For now, let's implement a robust search or default path.
                # Keysight often uses 'Waveforms' -> Channel -> 'RawData'

                # Looking for the channel group
                base_group = f.get("Waveforms")
                if not base_group:
                    # Fallback or root search
                    base_group = f

                channel_group = base_group.get(self.channel)
                if not channel_group:
                    # Try to find channel by partial match or taking the first one
                    keys = list(base_group.keys())
                    if keys:
                        channel_group = base_group[keys[0]]
                    else:
                        raise ValueError(
                            f"Could not find channel group '{self.channel}' or any valid group."
                        )

                # Assuming the first element contains the data
                element_key = (
                    list(channel_group.keys())[0] if channel_group.keys() else None
                )
                if not element_key:
                    raise ValueError("Empty channel group.")

                data_group = channel_group[element_key]

                # Check for RawData or look for any dataset
                if isinstance(data_group, h5py.Dataset):
                    raw_data = data_group[:]
                    attrs = dict(data_group.attrs)
                else:
                    # It's a group, look for dataset
                    dataset = None
                    if "RawData" in data_group:
                        dataset = data_group["RawData"]
                    else:
                        # Find first dataset
                        for key in data_group.keys():
                            if isinstance(data_group[key], h5py.Dataset):
                                dataset = data_group[key]
                                break

                    if dataset is None:
                        raise ValueError(
                            f"No dataset found in group {data_group.name}. Keys: {list(data_group.keys())}"
                        )

                    raw_data = dataset[:]
                    attrs = dict(data_group.attrs)
                    # Merge dataset attrs if any? usually on group or dataset.
                    attrs.update(dict(dataset.attrs))

                # Metadata extraction
                x_inc = attrs.get("XIncrement", 1.0)
                x_org = attrs.get("XOrigin", 0.0)

                points = len(raw_data)

                time = np.linspace(x_org, x_org + (points - 1) * x_inc, points)

                return NMRData(time=time, signal=raw_data, metadata=attrs)

        except Exception as e:
            raise RuntimeError(f"Failed to load Keysight file {file_path}: {e}")
