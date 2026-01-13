import numpy as np

from nmr_analysis.analysis.processing import extract_echo_train
from nmr_analysis.core.types import NMRData


def test_extract_echo_train_synthetic():
    # Create a synthetic echo train
    time = np.linspace(0, 1000, 10000)
    signal = np.zeros_like(time)

    # 5 peaks at 100, 300, 500, 700, 900
    peak_locs = [100, 300, 500, 700, 900]
    expected_amps = [10.0, 8.0, 6.4, 5.12, 4.096]  # decay

    for loc, amp in zip(peak_locs, expected_amps):
        # Gaussian shape
        signal += amp * np.exp(-0.5 * ((time - loc) / 10.0) ** 2)

    data = NMRData(time=time, signal=signal)

    times, amps, _, _ = extract_echo_train(data)

    # Expect absolute times (extract_echo_train no longer trims)
    expected_times = [100, 300, 500, 700, 900]

    assert len(times) == 5
    assert np.allclose(times, expected_times, atol=1.0)
    assert np.allclose(amps, expected_amps, atol=0.1)


if __name__ == "__main__":
    try:
        test_extract_echo_train_synthetic()
        print("test_extract_echo_train_synthetic PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise
