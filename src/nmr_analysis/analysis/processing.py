from typing import Tuple
import numpy as np

from nmr_analysis.core.types import NMRData


def extract_peak_amplitude(data: NMRData, method: str = "max_abs") -> float:
    """
    Extract the signal amplitude from an NMR trace.

    Args:
        data: NMRData object.
        method: 'max', 'min', 'max_abs', 'integral'.

    Returns:
        Amplitude value.
    """
    signal = data.signal

    if method == "max":
        return np.max(signal)
    elif method == "min":
        return np.min(signal)
    elif method == "max_abs":
        return np.max(np.abs(signal))
    elif method == "integral":
        return np.trapz(np.abs(signal), data.time)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_echo_train(
    data: NMRData, min_distance: int = 100, threshold_rel: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract peaks from a CPMG echo train.

    Args:
        data: NMRData object.
        min_distance: Minimum number of samples between peaks.
        threshold_rel: Relative threshold (0-1) of max peak to consider.

    Returns:
        Tuple of (peak_times, peak_amplitudes)
    """
    from scipy.signal import find_peaks

    signal = np.abs(data.signal)
    max_sig = np.max(signal)
    height = max_sig * threshold_rel

    # find_peaks returns indices
    peaks, _ = find_peaks(signal, height=height, distance=min_distance)

    peak_times = data.time[peaks]
    peak_amps = signal[peaks]

    return peak_times, peak_amps


def get_delay_from_metadata(data: NMRData) -> float:
    """
    Attempt to extract the delay parameter (tau) from metadata.
    This is highly specific to how data is saved.
    For now, return a placeholder or check common keys.
    """
    # Placeholder: user might need to supply this or regex the filename (not available here directly yet)
    # Check for 'tau', 'delay', 'wait', etc.
    for key in ["tau", "delay", "wait_time", "interval"]:
        if key in data.metadata:
            return float(data.metadata[key])
    return 0.0


def parse_time_from_filename(filename: str) -> float:
    """
    Extract time value from filename.
    Expected formats: '10ms', '0.5s', 'data_100us'.
    Returns time in seconds.
    """
    import re

    # Match patterns like 10ms, 10.5us, etc.
    # Simple regex searching for number followed by unit
    match = re.search(r"([\d\.]+)\s*(ms|us|s|ns)", filename)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == "s":
            return val
        if unit == "ms":
            return val * 1e-3
        if unit == "us":
            return val * 1e-6
        if unit == "ns":
            return val * 1e-9

    # Fallback: look for just a floating point number and assume seconds
    # e.g. "0_005.HDF5" -> 0.005
    # Replace underscores with dots if they are likely separators for decimals
    # But be careful about ID numbers.
    # Given "0_005", it looks like 0.005.

    clean_name = filename.replace(".HDF5", "").replace(".h5", "").replace(".hdf5", "")
    # Try to parse the whole name as a number if possible, replacing _ with .
    try:
        # heuristic: if it looks like X_XXX it might be X.XXX
        # or just try matching a float in the string
        # Let's try finding a number in the string again
        match_num = re.search(r"(\d+)[_.](\d+)", clean_name)
        if match_num:
            # Construct float
            v = float(f"{match_num.group(1)}.{match_num.group(2)}")
            return v

        # Or just a simple int/float match
        match_simple = re.search(r"([\d\.]+)", clean_name)
        if match_simple:
            return float(match_simple.group(1))

    except Exception:
        pass

    return 0.0
