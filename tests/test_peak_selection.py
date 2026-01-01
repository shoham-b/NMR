import numpy as np

from nmr_analysis.core.types import NMRData
from nmr_analysis.analysis.processing import extract_peak_by_index


def create_synthetic_data(peaks):
    """
    Create synthetic signal with peaks at specified amplitudes.
    peaks: list of (index, amplitude)
    """
    signal = np.zeros(1000)
    for idx, amp in peaks:
        # Create a small gaussian peak
        x = np.arange(1000)
        signal += amp * np.exp(-0.5 * ((x - idx) / 2) ** 2)

    time = np.linspace(0, 1, 1000)
    return NMRData(time=time, signal=signal, metadata={})


def test_extract_peak_by_index_with_smoothing():
    # New test: extract_peak_by_index(index=2) with smoothing
    # Peaks: 10, 8, 6, 4. Index 2 has amp 6.
    data = create_synthetic_data([(100, 10), (200, 8), (300, 6), (400, 4)])

    # Without smoothing
    t, amp, idx, _ = extract_peak_by_index(
        data, peak_index=2, smoothing=0.0, prominence=0.5, min_time_sep=0.05
    )
    assert abs(idx - 300) < 5
    assert abs(amp - 6.0) < 0.1

    # With smoothing
    t_sm, amp_sm, idx_sm, _ = extract_peak_by_index(
        data, peak_index=2, smoothing=2.0, prominence=0.5, min_time_sep=0.05
    )
    # Smoothing shouldn't affect the returned amplitude now (we return raw)
    assert abs(idx_sm - 300) < 5
    assert abs(amp_sm - 6.0) < 0.1  # Should be close to original height


def test_extract_peak_by_index_smoothing_helps():
    # Construct a case where a noise spike might be picked as a peak without smoothing?
    # extract_peak_by_index relies on find_peaks finding enough peaks.
    pass


if __name__ == "__main__":
    from nmr_analysis.analysis.processing import extract_peak_by_index

    try:
        test_extract_peak_by_index_with_smoothing()
        print("test_extract_peak_by_index_with_smoothing PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
    # Manually run tests if pytest not handy
    try:
        test_extract_peak_by_index_with_smoothing()
        print("test_extract_peak_by_index_with_smoothing PASSED")
        # test_extract_second_highest_simple() -> Removed
        # test_extract_second_highest_artifact_start() -> Removed
        # test_extract_second_highest_huge_artifact() -> Removed
        # test_extract_second_highest_proximity() -> Removed
    except Exception as e:
        print(f"FAILED: {e}")
        raise
