import numpy as np
import pytest
from nmr_analysis.analysis.processing import find_peaks_t1_t2
from nmr_analysis.core.types import NMRData, ExperimentType


def create_synthetic_data(peaks, offset=0.0):
    """
    peaks: list of (index, amplitude)
    """
    time = np.linspace(0, 1000, 1000)
    signal = np.zeros_like(time) + offset
    for idx, amp in peaks:
        # Simple Gaussian spikes
        # Width very narrow to avoid overlap
        signal += amp * np.exp(-0.5 * ((np.arange(1000) - idx) / 2.0) ** 2)

    return NMRData(time=time, signal=signal, metadata={}, experiment_type=None)


def test_t1_logic_basic():
    # T1 Logic: DC Offset removal, Absolute signal, First and Last.
    # Create data with DC offset 10.
    # Peaks at 100 (Amp 50), 300 (Amp 20), 500 (Amp 40).
    # Expected: First (100) and Last (500).
    # Amplitudes should be relative to offset?
    # Function returns detection_signal values which are abs(signal - median).

    data = create_synthetic_data([(100, 50), (300, 20), (500, 40)], offset=10.0)
    data.experiment_type = ExperimentType.T1

    p1_idx, tau, amp, info = find_peaks_t1_t2(
        data, experiment_type=ExperimentType.T1, smoothing=0
    )

    # P1 should be 100
    assert abs(p1_idx - 100) < 5
    # Fit idx should be 500 (Last)
    assert abs(info["fit_idx"] - 500) < 5

    # Amplitude check
    # Signal at 500 is 40 + 10 = 50. Median is 10. Corrected is 40.
    assert abs(amp - 40.0) < 1.0


def test_t2_logic_ratio_select_p3():
    # T2 Logic: Select P3 if P3/P2 >= 0.6
    # P1 is max.
    # P1=100 (Amp 100)
    # P2=40 (Amp 40)
    # P3=32 (Amp 32) -> Ratio 0.8 -> Select P3.

    data = create_synthetic_data([(100, 100), (300, 40), (500, 32)])
    data.experiment_type = ExperimentType.T2

    p1_idx, tau, amp, info = find_peaks_t1_t2(
        data, experiment_type=ExperimentType.T2, smoothing=0
    )

    assert abs(p1_idx - 100) < 5
    assert abs(info["fit_idx"] - 500) < 5  # Selected P3


def test_t2_logic_ratio_select_p2():
    # T2 Logic: Select P2 if P3/P2 < 0.6
    # P1=100 (Amp 100)
    # P2=40 (Amp 40)
    # P3=20 (Amp 20) -> Ratio 0.5 -> Select P2.

    data = create_synthetic_data([(100, 100), (300, 40), (500, 20)])
    data.experiment_type = ExperimentType.T2

    p1_idx, tau, amp, info = find_peaks_t1_t2(
        data, experiment_type=ExperimentType.T2, smoothing=0
    )

    assert abs(p1_idx - 100) < 5
    assert abs(info["fit_idx"] - 300) < 5  # Selected P2


def test_t2_logic_only_two_peaks():
    # Only P1 and P2
    data = create_synthetic_data([(100, 100), (300, 50)])
    data.experiment_type = ExperimentType.T2

    p1_idx, tau, amp, info = find_peaks_t1_t2(
        data, experiment_type=ExperimentType.T2, smoothing=0
    )

    assert abs(p1_idx - 100) < 5
    assert abs(info["fit_idx"] - 300) < 5


if __name__ == "__main__":
    test_t1_logic_basic()
    test_t2_logic_ratio_select_p3()
    test_t2_logic_ratio_select_p2()
    test_t2_logic_only_two_peaks()
    print("All tests passed!")
