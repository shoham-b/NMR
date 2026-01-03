def test_extract_echo_train_monotonic_filter():
    from nmr_analysis.analysis.processing import extract_echo_train
    from nmr_analysis.core.types import NMRData
    import numpy as np

    # Create synthetic data with a non-monotonic peak (dip)
    # Peaks: 100, 20 (dip), 80.
    # Logic should keep 80, discard 20 because 20 < 80 (next peak is higher).
    # Then keep 100 because 100 > 80.
    # Result should be [100, 80].

    time = np.linspace(0, 10, 1000)
    signal = np.zeros_like(time)

    # Peaks at indices 100, 200, 300
    signal[100] = 100.0
    signal[200] = 20.0  # Dip
    signal[300] = 80.0

    data = NMRData(time=time, signal=signal)

    # Use minimal smoothing to avoid shifting/blurring too much
    # Use min_distance small enough to resolve them
    peaks, amps = extract_echo_train(
        data, min_distance=10, smoothing=0.0, min_height=5.0
    )

    # Check that 20.0 is NOT in amps
    # And 100, 80 ARE.
    assert len(amps) == 2
    assert 100.0 in amps
    assert 80.0 in amps
    assert 20.0 not in amps
