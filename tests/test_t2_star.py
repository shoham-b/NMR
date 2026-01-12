import numpy as np
import pytest
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.models import t2_decay_model
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
    decay_time = time[peak_idx:]
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
    tail_length = n_samples - peak_idx
    expected_start_offset = 0  # int(tail_length * 0.0)
    expected_start_idx = peak_idx + expected_start_offset

    print(f"Computed start_index: {result.metadata.get('start_index')}")
    print(f"Expected start_index: {expected_start_idx}")

    assert result.metadata["start_index"] == expected_start_idx

    # Check that fit_curve contains NaNs before start_idx
    assert np.all(np.isnan(result.fit_curve[:expected_start_idx]))
    # And values after
    assert not np.all(np.isnan(result.fit_curve[expected_start_idx:]))


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
    Test that fitting fails gracefully (returns result with error/NaNs)
    instead of raising RuntimeError when there are fewer data points than parameters (3).
    """
    time = np.linspace(0, 10, 100)
    signal = np.zeros_like(time)

    # Create a scenario where peak finding + trimming leaves < 3 points
    # e.g., Peak at the very end
    peak_idx = 98
    signal[peak_idx] = 10.0

    # This will lead to start_idx around 98.
    # length is 100.
    # data points: 98, 99. Length = 2.
    # Parameters needed: 3 (M0, T2, offset).

    data = NMRData(time=time, signal=signal, metadata={})

    # Should not raise exception
    result = Fitter.fit_t2_star(data)

    # Should be a failed fit result
    assert result.r_squared == 0.0
    assert "Insufficient data" in str(
        result.metadata.get("error", "")
    ) or "Fit Failed" in str(result.metadata.get("error", ""))


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
