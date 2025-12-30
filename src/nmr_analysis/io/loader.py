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
                # Updated per user request
                x_unit_path = "/__BV_Dataset__Data__/data_chan1_capture1/___BV___CUSTOM_LONG_METADATA__XUnits"
                y_unit_path = "/__BV_Dataset__Data__/data_chan1_capture1/___BV___CUSTOM_LONG_METADATA__YUnits"

                try:
                    for path, key in [
                        (x_unit_path, "time_unit"),
                        (y_unit_path, "signal_unit"),
                    ]:
                        if path in f:
                            unit_data = f[path][()]
                            # Handle numpy array or scalar
                            if hasattr(unit_data, "flatten"):
                                unit_data = unit_data.flatten()
                                if len(unit_data) > 0:
                                    unit_data = unit_data[0]

                            # Handle bytes to string decoding
                            if isinstance(unit_data, (bytes, np.bytes_)):
                                unit_data = unit_data.decode("utf-8")

                            attrs[key] = str(unit_data)
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


class CSVLoader:
    """Loader for CSV files with specific schema."""

    def __init__(self, channel: str = "Channel 1"):
        self.channel = channel

    def load(self, file_path: Path) -> NMRData:
        """
        Load data from a CSV file.

        Schema:
        - Col 0: Metadata Name
        - Col 1: Metadata Value
        - Col 2: Empty
        - Col 3: Time
        - Col 4: Channel 1 (column_5)
        - Col 5: Channel 2 (column_6)

        The first two rows of data columns (3, 4, 5) are headers and should be skipped for data.
        Metadata might be present in the first few rows.
        """
        import polars as pl

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # infer_schema_length=0 forces all columns to be read as String (Utf8)
            df = pl.read_csv(
                file_path,
                has_header=False,
                truncate_ragged_lines=True,
                infer_schema_length=0,
            )

            # Metadata extraction (same as before)
            meta_names = df["column_1"]
            meta_values = df["column_2"]
            metadata = {}
            for k, v in zip(meta_names, meta_values):
                if k is not None and v is not None:
                    k_str = str(k).strip()
                    if k_str:
                        v_val = v
                        try:
                            v_val = float(v)
                        except (ValueError, TypeError):
                            pass
                        metadata[k_str] = v_val

            # Data Extraction
            # first two rows are headers for data columns
            data_slice = df.slice(2)

            time_col = data_slice["column_4"].cast(pl.Float64, strict=False)

            # Select signal column based on channel
            # Polars default names: column_N (1-based index)
            # Channel 1 -> Col 4 (0-based) -> column_5
            # Channel 2 -> Col 5 (0-based) -> column_6

            # Simple mapping or parsing logic
            target_col = "column_5"  # Default to Channel 1
            if "2" in self.channel:
                target_col = "column_6"

            if target_col not in df.columns:
                # Fallback if file doesn't have that many columns?
                # Should raise error or fallback?
                # If user asks for Ch2 but it doesn't exist, error is appropriate.
                pass

            signal_col = data_slice[target_col].cast(pl.Float64, strict=False)

            # Filter valid data
            mask = time_col.is_not_null() & signal_col.is_not_null()

            time = time_col.filter(mask).to_numpy()
            signal = signal_col.filter(mask).to_numpy()

            return NMRData(time=time, signal=signal, metadata=metadata)

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file {file_path}: {e}")


def get_loader(file_path: Path, channel: str = "Channel 1"):
    """
    Factory function to get the correct loader based on file extension.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return CSVLoader(channel=channel)
    elif suffix in (".h5", ".hdf5"):
        return KeysightLoader(channel=channel)
    else:
        raise ValueError(
            f"Unsupported file extension: {suffix} (Expected .h5, .hdf5, or .csv)"
        )
