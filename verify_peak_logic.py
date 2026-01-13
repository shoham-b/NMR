import numpy as np
from nmr_analysis.core.types import NMRData, ExperimentType
from nmr_analysis.analysis.processing import find_peaks_t1_t2


def test_peak_selection():
    # Simulate an Echo Train
    # P1 at t=0, Amp=1.0 (Pulse)
    # P2 at t=10, Amp=0.8 (Echo 1, T2 approx 44)
    # P3 at t=20, Amp=0.5 (Echo 2, T2 approx 28 -> Faster decay)
    # P4 at t=30, Amp=0.7 (Noise Spike! High amp -> Slow decay relative to P1)

    # We want logic to pick P4 if it maximizes T2?
    # T2_implied(P2) = -10 / ln(0.8/1.0) = -10 / -0.223 = 44.8
    # T2_implied(P3) = -20 / ln(0.5/1.0) = -20 / -0.693 = 28.8
    # T2_implied(P4) = -30 / ln(0.7/1.0) = -30 / -0.356 = 84.2

    # "Longest Decay" -> P4 should be chosen.
    # Note: P1 is skipped as a candidate.

    time = np.linspace(0, 100, 1000)
    signal = np.zeros_like(time)

    # Create peaks
    idx1 = 0
    idx2 = 100
    idx3 = 200
    idx4 = 300

    signal[idx1] = 1.0  # P1
    signal[idx2] = 0.8  # P2
    signal[idx3] = 0.5  # P3
    signal[idx4] = 0.7  # P4 (Spike)

    data = NMRData(time=time, signal=signal)

    # Run find_peaks_t1_t2
    # We need to ensure find_peaks finds them.
    # min_height=0.1, distance=10

    p1, tau, amp, info = find_peaks_t1_t2(data, min_height=0.1, min_distance=10)

    print(f"Selected Peak Index: {info['fit_idx']}")
    print(f"Selected Amp: {amp}")
    print(f"All peaks: {info['all_peaks']}")

    # Expectation: fit_idx should be idx4 (300) because it implies slowest decay (T2=84)
    if info["fit_idx"] == idx4:
        print("SUCCESS: Picked P4 (Slowest Decay)")
    elif info["fit_idx"] == idx2:
        print("RESULT: Picked P2 (First valid echo)")
    else:
        print(f"RESULT: Picked {info['fit_idx']}")


if __name__ == "__main__":
    test_peak_selection()
