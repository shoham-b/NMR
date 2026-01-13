from nmr_analysis.io.loader import get_loader
from nmr_analysis.analysis.processing import preprocess_data
import pathlib
import numpy as np

path = pathlib.Path(r"H:\My Drive\Lab C\NMR\week4.2\methanol\T2\100.csv")
L = get_loader(path)
d = L.load(path)
print(f"Signal length: {len(d.signal)}")
print(f"Time range: {d.time[0]} to {d.time[-1]} seconds")
print(f"Sample rate: {len(d.signal) / (d.time[-1] - d.time[0]):.0f} Hz")

pd, tau, amp, pi = preprocess_data(d)
print(f"\nProcessed sig len: {len(pd.signal)}")
print(f"Processed time range: {pd.time[0]} to {pd.time[-1]} seconds")
fit_idx = pi.get("fit_idx", None)
print(f"\nfit_idx: {fit_idx}")
print(f"All peaks: {pi.get('all_peaks', [])}")

if fit_idx is not None and fit_idx < len(pd.time):
    print(f"Time at fit_idx: {pd.time[fit_idx]} seconds")
    print(f"Amp at fit_idx: {np.abs(pd.signal[fit_idx])}")
    print(
        f"Max signal amp in first 0.1s: {np.max(np.abs(pd.signal[: int(0.1 / (pd.time[1] - pd.time[0]))]))}"
    )
