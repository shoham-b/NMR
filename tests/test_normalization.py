"""
Tests for NMR data preprocessing.

These tests verify the basic preprocessing contract:
- Data structure preservation
- Metadata preservation
- No data loss during preprocessing
"""

import numpy as np
from nmr_analysis.core.types import NMRData
from nmr_analysis.analysis.processing import preprocess_data


def test_preprocess_data_no_data_loss():
    """Test that preprocessing does not lose data points."""
    time = np.linspace(0, 10, 100)
    signal = np.zeros(100)
    signal[10] = 6.0
    signal[9] = 1.0
    signal[11] = 1.0

    data = NMRData(time=time, signal=signal)

    processed, t_orig, amp_orig, info = preprocess_data(
        data, smoothing=0, min_height=1.0
    )

    # No data loss
    assert len(processed.signal) == 100
    assert len(processed.time) == 100

    # Signal values preserved
    assert np.allclose(processed.signal, signal)

    # Returns valid amplitude
    assert amp_orig > 0


def test_preprocess_data_metadata_preserved():
    """Test that metadata is preserved during preprocessing."""
    time = np.linspace(0, 10, 100)
    signal = np.zeros(100)
    signal[50] = 6.0
    data = NMRData(time=time, signal=signal, metadata={"key": "val"})

    processed, _, _, _ = preprocess_data(data, smoothing=0, min_height=1.0)
    assert processed.metadata["key"] == "val"


def test_preprocess_returns_peak_info():
    """Test that preprocessing returns peak information."""
    time = np.linspace(0, 10, 100)
    signal = np.zeros(100)
    signal[10] = 10.0
    signal[50] = 8.0
    signal[80] = 6.0

    data = NMRData(time=time, signal=signal)

    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Returns peak info dict
    assert isinstance(info, dict)

    # Has some peak-related keys
    assert len(info) > 0

    # Returns valid tau (non-negative)
    assert tau >= 0

    # Returns valid amplitude
    assert amp > 0


def test_preprocess_handles_flat_signal():
    """Test that preprocessing handles a flat signal gracefully."""
    time = np.linspace(0, 10, 100)
    signal = np.ones(100) * 0.1  # Flat low signal

    data = NMRData(time=time, signal=signal)

    # Should not raise
    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=0.01)

    # Returns same length
    assert len(processed.time) == len(time)
    assert len(processed.signal) == len(signal)
