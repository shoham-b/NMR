import sys
from pathlib import Path
import numpy as np

# Paths
NMRMINE_PATH = Path(r"c:\Users\shoha\git\NMRMINE")
NMR_PATH = Path(r"c:\Users\shoha\git\NMR")
DATA_FILE = NMRMINE_PATH / "data/T2/100.csv"

# 1. Setup NMRMINE import
sys.path.append(str(NMRMINE_PATH / "src"))
try:
    from t2_multiple_analysis import analyze_multiple_echo_file

    print("Successfully imported analyze_multiple_echo_file from NMRMINE")
except ImportError as e:
    print(f"Failed to import from NMRMINE: {e}")
    sys.exit(1)

# 2. Setup NMR import
sys.path.append(str(NMR_PATH / "src"))
try:
    from nmr_analysis.analysis.fitting import Fitter

    print("Successfully imported Fitter from NMR")
except ImportError as e:
    print(f"Failed to import from NMR: {e}")
    sys.exit(1)


def run_bridge():
    print(f"Analyzing {DATA_FILE}...")

    # Run NMRMINE logic
    # We need to capture the internal data (peaks), but analyze_multiple_echo_file only returns a summary dict.
    # However, it prints "Selected Peak Times" and "Selected Peak Amps" in debug mode or we might need to modify it?
    # Wait, analyze_multiple_echo_file in NMRMINE/src/t2_multiple_analysis.py returns:
    # { "file": ..., "T2": ..., "error": ... }
    # It does NOT return peak arrays.
    # I need to modify NMRMINE or extract the logic.
    # modifying the imported function on the fly is hard.
    # But wait, looking at the code I read earlier:

    # 86:     # 5. Fit
    # ...
    # 139:     return {
    # 140:         "file": file_path.name,
    # 141:         "T2": T2_fit,
    # 142:         "error": T2_err
    # 143:     }

    # It does NOT return the arrays. This is an issue.
    # I will need to "extract" the logic.
    # Since I cannot easily modify the return of the existing function without editing the file,
    # and the user wants to "use the peak finding", I should probably copy the logic into this script
    # to demonstrate (and eventually into NMR repo).

    # Let's verify we can load the data using NMRMINE's loader first, as that's part of the "logic".
    from loader import get_loader

    loader = get_loader(DATA_FILE)
    data = loader.load(DATA_FILE)

    # --- Replicating NMRMINE Logic ---
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    # 1. Find Global Max (Time Zero)
    max_idx = np.argmax(data.signal)
    max_time = data.time[max_idx]
    max_val = data.signal[max_idx]

    # 2. Shift Time and Slice
    time_shifted = data.time - max_time
    mask = time_shifted >= 0
    t_slice = time_shifted[mask]
    s_slice = data.signal[mask]

    # 3. Find All Potential Peaks
    peaks_all, _ = find_peaks(
        s_slice, height=0.05 * max_val, distance=50, prominence=0.05 * max_val
    )

    if 0 not in peaks_all:
        peaks_all = np.insert(peaks_all, 0, 0)

    # 4. Filter for Monotonic Decay
    valid_indices = []
    max_amp_so_far = -1.0

    for idx in reversed(peaks_all):
        amp = s_slice[idx]
        if amp > max_amp_so_far:
            valid_indices.append(idx)
            max_amp_so_far = amp

    valid_indices = sorted(valid_indices)
    peak_times = t_slice[valid_indices]
    peak_amps = s_slice[valid_indices]

    print(f"NMRMINE Logic Found {len(peak_times)} peaks.")
    print(f"Peak Times: {peak_times[:5]}...")
    print(f"Peak Amps: {peak_amps[:5]}...")

    # --- Feed to NMR Logic ---
    fitter = Fitter()
    # Fitter.fit_t2 takes (delays, amplitudes)

    print("Fitting using NMR Fitter...")
    # Note: fit_t2 returns params, fit_curve, residuals, r_squared, param_errors
    # params is likely (A, T2) or similar?
    # Let's check fitting.py again if needed, but usually it returns params.

    params, fit_curve, residuals, r2, errors = fitter.fit_t2(peak_times, peak_amps)

    print(f"NMR Fitter Result:")
    print(f"Params: {params}")
    print(f"R2: {r2}")


run_bridge()
