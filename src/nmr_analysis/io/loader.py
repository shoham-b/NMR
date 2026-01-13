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
                infer_schema_length=None,
                truncate_ragged_lines=False,
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


class OscilloscopeLoader:
    """Loader for Oscilloscope CSV files."""

    def __init__(self, channel: str = "Channel 1"):
        self.channel = channel

    def load(self, file_path: Path) -> NMRData:
        """
        Load data from an Oscilloscope CSV file.
        Supports two formats:
        1. Side-by-Side (DSOX1204G): Metadata in cols 0,1; Data in cols 3,4. Header at row 0.
        2. Vertical (Legacy): Metadata rows, then blank line, then Header.
        """

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect Format based on first few lines
        is_side_by_side = False
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                # Check for characteristic columns of side-by-side
                if "Time (s)" in line and "(VOLT)" in line:
                    is_side_by_side = True
                    break

        if is_side_by_side:
            return self._load_side_by_side(file_path)
        else:
            return self._load_vertical(file_path)

    def _load_side_by_side(self, file_path: Path) -> NMRData:
        import polars as pl
        import csv

        # Load with Polars (Header at row 0 or deeper)
        try:
            # Find header row first
            skip_rows = 0
            with open(file_path, "r", encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if "Time" in line and "(s)" in line and "(VOLT)" in line:
                        skip_rows = i
                        break

            df = pl.read_csv(
                file_path,
                has_header=True,
                skip_rows=skip_rows,
                infer_schema_length=1000,
                truncate_ragged_lines=True,
            )

            # Metadata (reload to be safe from top)
            metadata = {}
            with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        key = row[0].strip()
                        val = row[1].strip()
                        if key:
                            if key.endswith(":"):
                                key = key[:-1]
                            metadata[key] = val

            # Data
            # Find Time Column
            time_cols = [c for c in df.columns if "Time" in c and "(s)" in c]
            if not time_cols:
                time_cols = [c for c in df.columns if "Time" in c]
            if not time_cols:
                raise ValueError("Time column not found")
            time_col_name = time_cols[0]

            # Find Signal Column
            # Target Channel
            target_ch = "1"
            if "2" in self.channel:
                target_ch = "2"

            # Look for explicit "{ch} (VOLT)"
            sig_cols = [c for c in df.columns if f"{target_ch} (VOLT)" in c]

            if not sig_cols:
                # If specifically looking for Ch 2 and not found, maybe valid error.
                # BUT if user says "Channel 2" but file only has Ch 1, and user implies "just load data",
                # we might need to fallback?
                # User default in commands.py is "Channel 2". If file is "Channel 1" only, that's a problem.
                # Let's try to fallback to ANY "VOLT" column if specific one missing?
                # No, that's dangerous. Better to fail or warn.
                # However, for robustness, if only 1 VOLT column exists, maybe use it?
                volt_cols = [c for c in df.columns if "(VOLT)" in c]
                if len(volt_cols) == 1:
                    sig_cols = volt_cols
                else:
                    raise ValueError(
                        f"Signal column for {self.channel} not found in {df.columns}"
                    )

            sig_col_name = sig_cols[0]

            time = df[time_col_name].cast(pl.Float64, strict=False)
            signal = df[sig_col_name].cast(pl.Float64, strict=False)

            mask = time.is_not_null() & signal.is_not_null()
            return NMRData(
                time=time.filter(mask).to_numpy(),
                signal=signal.filter(mask).to_numpy(),
                metadata=metadata,
            )

        except Exception as e:
            raise RuntimeError(f"Side-by-side load failed: {e}")

    def _load_vertical(self, file_path: Path) -> NMRData:
        import polars as pl
        import csv

        metadata = {}
        skip_rows = 0
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    skip_rows = i + 1
                    break
                if len(row) >= 1:
                    line = row[0]
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip()
                    elif len(row) >= 2:
                        metadata[row[0].strip()] = row[1].strip()

        try:
            df = pl.read_csv(
                file_path,
                skip_rows=skip_rows,
                has_header=True,
                infer_schema_length=1000,
                truncate_ragged_lines=True,
            )

            # Map columns logic (reuse or simplify)
            time_cols = [c for c in df.columns if "Time" in c]
            if not time_cols:
                raise ValueError("Time column not found")
            time_col_name = time_cols[0]

            target_ch = "1"
            if "2" in self.channel:
                target_ch = "2"

            # Check for "1 (VOLT)" or just "1" or "Channel 1"
            sig_cols = [c for c in df.columns if f"{target_ch} (VOLT)" in c]
            if not sig_cols:
                sig_cols = [c for c in df.columns if c.strip() == target_ch]

            if not sig_cols:
                # Fallback
                volt_cols = [c for c in df.columns if "(VOLT)" in c]
                if len(volt_cols) == 1:
                    print(
                        f"WARNING: Requested {self.channel} ({target_ch}) not found. using {volt_cols[0]}"
                    )
                    sig_cols = volt_cols
                else:
                    raise ValueError(
                        f"Signal column for {self.channel} not found in {df.columns}"
                    )

            sig_col_name = sig_cols[0]

            time = df[time_col_name].cast(pl.Float64, strict=False)
            signal = df[sig_col_name].cast(pl.Float64, strict=False)
            mask = time.is_not_null() & signal.is_not_null()
            return NMRData(
                time=time.filter(mask).to_numpy(),
                signal=signal.filter(mask).to_numpy(),
                metadata=metadata,
            )

        except Exception as e:
            raise RuntimeError(f"Vertical load failed: {e}")


def get_loader(file_path: Path, channel: str = "Channel 1"):
    """
    Factory function to get the correct loader based on file extension.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        # Sniff content to distinguish between generic CSV and Oscilloscope CSV
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                # Read header lines
                header_lines = []
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    header_lines.append(line)

                content_chunk = "".join(header_lines)

                # Check for New Oscilloscope format
                # User specified: Model: Oscilloscope DSOX1204G
                # Relaxed check: Just Model and DSOX1204G
                if "Model" in content_chunk and "DSOX1204G" in content_chunk:
                    return OscilloscopeLoader(channel=channel)

        except Exception:
            pass

        return CSVLoader(channel=channel)
    elif suffix in (".h5", ".hdf5"):
        return KeysightLoader(channel=channel)
    else:
        raise ValueError(
            f"Unsupported file extension: {suffix} (Expected .h5, .hdf5, or .csv)"
        )
