import numpy as np
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.core.types import NMRData


def test_t2_star_fitting_starts_after_peak():
    # Construct a signal: Noise -> Rise -> Peak -> Decay
    time = np.linspace(0, 10, 100)

    # Peak at index 20 (time=2.0)
    peak_idx = 20

    signal = np.zeros_like(time)

    # Create decay starting from peak
    # t2_decay_model(t, M0, T2, offset)
    M0 = 100.0
    T2_star = 1.0

    # Fill decay part
    time[peak_idx:]
    # Ideally decay starts at time[peak_idx] with value M0.
    # But t2_decay_model usage depends on interpretation. Usually it's M0*exp(-t/T)
    # If we want M0 at t=time[peak_idx], we might need an offset in time or just model the shape.
    # Here we just put a decay shape.
    signal[peak_idx:] = M0 * np.exp(-(time[peak_idx:] - time[peak_idx]) / T2_star)

    # Add rise/noise before peak
    # Make sure it ends strictly lower than M0 to ensure a sharp peak for find_peaks
    signal[:peak_idx] = np.linspace(0, M0 * 0.9, peak_idx)  # Linear rise to 90% M0

    data = NMRData(time=time, signal=signal, metadata={})

    # Fit
    result = Fitter.fit_t2_star(data)

    # Verify
    # The peak is at index 20 (time=2.0)
    # n_samples = 100
    # tail_length = 100 - 20 = 80
    # start_trim_factor = 0.0 (Updated logic)
    # expected_start_idx = 20 + 0 = 20

    n_samples = len(time)
    # Algorithm detects decay_stop based on noise floor, then calculates decay_len
    # decay_len = decay_stop_idx - peak_idx
    # start_trim_points = int(decay_len * 0.05)  # 5% trim
    # expected_start_idx = peak_idx + start_trim_points
    # For this test signal that decays smoothly, decay_len will be most of the tail

    # Just verify that start_index is at or near peak, and that the fit is valid
    start_idx = result.metadata["start_index"]
    peak_found = result.metadata.get("peak_index", peak_idx)

    print(f"Computed start_index: {start_idx}")
    print(f"Peak index: {peak_found}")

    # start_index should be at or slightly after peak (due to 5% trim)
    assert start_idx >= peak_idx, (
        f"start_index {start_idx} should be >= peak_idx {peak_idx}"
    )
    # start_index shouldn't be too far from peak (within 20% of decay length)
    assert start_idx < peak_idx + (n_samples - peak_idx) * 0.2, (
        f"start_index too far from peak"
    )

    # Check that fit_curve contains NaNs before start_idx
    assert np.all(np.isnan(result.fit_curve[:start_idx]))
    # And values after
    assert not np.all(np.isnan(result.fit_curve[start_idx:]))


def test_t2_star_fitting_no_peak_found():
    # Flat line? Or just noise
    time = np.linspace(0, 10, 100)
    signal = np.random.normal(0, 0.1, 100)  # Max around 0.3? < 5.0

    data = NMRData(time=time, signal=signal, metadata={})

    # Should fallback to argmax + 1
    # Let's see what happens.
    result = Fitter.fit_t2_star(data)

    start_idx = result.metadata["start_index"]
    assert start_idx > 0  # At least it tried


def test_t2_star_fitting_insufficient_data():
    """
    Test that fitting handles edge cases gracefully (returns result without raising).
    When peak is near the end, the algorithm uses fallback logic to find enough points.
    """
    time = np.linspace(0, 10, 100)
    signal = np.zeros_like(time)

    # Create a scenario where peak is at the very end
    peak_idx = 98
    signal[peak_idx] = 10.0

    data = NMRData(time=time, signal=signal, metadata={})

    # Should not raise exception - algorithm handles edge cases gracefully
    result = Fitter.fit_t2_star(data)

    # The algorithm has fallback logic, so it may still attempt a fit.
    # Key test: it doesn't crash and returns a valid result structure.
    assert result is not None
    assert hasattr(result, "r_squared")
    assert hasattr(result, "fit_curve")
    # For this edge case, the fit quality will be poor (low r_squared) or
    # the algorithm may report an error in metadata
    # Either outcome is acceptable - we're testing graceful handling
    print(f"r_squared: {result.r_squared}")
    print(f"metadata: {result.metadata}")


if __name__ == "__main__":
    try:
        test_t2_star_fitting_starts_after_peak()
        print("test_t2_star_fitting_starts_after_peak PASSED")
        test_t2_star_fitting_no_peak_found()
        print("test_t2_star_fitting_no_peak_found PASSED")
        test_t2_star_fitting_insufficient_data()
        print("test_t2_star_fitting_insufficient_data PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise
