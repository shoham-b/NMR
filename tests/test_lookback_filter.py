import numpy as np
import pytest
from nmr_analysis.analysis.processing import filter_peaks_monotonic_reverse


def test_reverse_monotonic_perfect_decay():
    """
    Test standard decay: [100, 90, 80]
    Reverse: 80 (Keep, Max=80) -> 90 (>80, Keep, Max=90) -> 100 (>90, Keep, Max=100).
    Result: [100, 90, 80].
    """
    indices = np.array([0, 1, 2])
    amps = np.array([100.0, 90.0, 80.0])

    filtered_indices = filter_peaks_monotonic_reverse(indices, amps)
    expected_indices = np.array([0, 1, 2])

    np.testing.assert_array_equal(filtered_indices, expected_indices)


def test_reverse_monotonic_user_case():
    """
    Test user scenario: "3 descending and after them 1 ascending"
    Amps: [100, 90, 80, 70, 95]
    Reverse Iteration:
    1. 95 (Keep, Max=95)
    2. 70 (<95, Drop)
    3. 80 (<95, Drop)
    4. 90 (<95, Drop)
    5. 100 (>95, Keep, Max=100)

    Expected Result: [100, 95] -> Indices [0, 4]
    """
    indices = np.array([0, 1, 2, 3, 4])
    amps = np.array([100.0, 90.0, 80.0, 70.0, 95.0])

    filtered_indices = filter_peaks_monotonic_reverse(indices, amps)
    expected_indices = np.array([0, 4])

    np.testing.assert_array_equal(filtered_indices, expected_indices)


def test_reverse_monotonic_buildup():
    """
    Test anomalous buildup (bad T2 data?): [10, 20, 30]
    Reverse:
    1. 30 (Keep, Max=30)
    2. 20 (<30, Drop)
    3. 10 (<30, Drop)

    Result: [30] -> Index [2]
    """
    indices = np.array([0, 1, 2])
    amps = np.array([10.0, 20.0, 30.0])

    filtered_indices = filter_peaks_monotonic_reverse(indices, amps)
    expected_indices = np.array([2])

    np.testing.assert_array_equal(filtered_indices, expected_indices)
