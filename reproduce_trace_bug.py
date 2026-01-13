import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path("src").resolve()
sys.path.append(str(src_path))

from nmr_analysis.core.types import NMRData
from nmr_analysis.cli.commands import plot_stacked_traces


def run_verification():
    print("Verifying fix with ACTUAL plot_stacked_traces...")

    # Mock data
    t = np.linspace(0, 1, 100)
    d = NMRData(time=t, signal=t, metadata={})
    processed_data = NMRData(time=t, signal=t, metadata={})
    peak_info = {}

    # The tuple as constructed in _run_analysis
    # (processed_data, t_peak, amp, tau, peak_info, data_full, sort_val)
    trace_tuple = (processed_data, 0.0, 1.0, 0.5, peak_info, d, 0.5)

    raw_traces = [trace_tuple]

    try:
        # Pass a filepath to avoid plt.show() blocking
        plot_stacked_traces(
            raw_traces, filepath=Path("test_debug_plot.png"), show_fourier=False
        )
        print("Function executed and saved plot without crashing.")

    except Exception as e:
        print(f"FAILED with error: {e}")
        exit(1)


if __name__ == "__main__":
    run_verification()
