import numpy as np
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.models import j_modulated_t2


def test_j_modulated_fit_synthetic():
    """
    Test fitting of synthetic J-modulated Spin Echo data.
    """
    # 1. Generate Synthetic Data
    # J = 7 Hz (Ethanol approx)
    # T2 = 0.5 sec
    M0 = 100.0
    T2 = 0.5
    J = 7.0
    offset = 5.0

    # Delays from 1ms to 1s
    delays = np.linspace(0.001, 1.0, 50)

    # Perfect data
    y_true = j_modulated_t2(delays, M0, T2, J, offset)

    # Add noise with fixed seed for reproducibility
    np.random.seed(42)
    noise = np.random.normal(0, 1.0, len(delays))
    y_noisy = y_true + noise

    # 2. Fit
    # Guess J=6.0 to make it work for it
    results_dict, fit_curve, residuals, r2, errors, outlier_mask = (
        Fitter.fit_modulated_t2(delays, y_noisy, guess_J=6.0)
    )

    print("\nFit Results:")
    print(results_dict)

    # 3. Assertions
    assert r2 > 0.95, f"R2 {r2} is too low"

    # Check params
    # Allow 10% error due to noise
    assert abs(results_dict["M0"] - M0) < 10.0
    assert abs(results_dict["T2"] - T2) < 0.1
    assert abs(results_dict["J"] - J) < 0.5
    assert abs(results_dict["offset"] - offset) < 2.0

    # Check J specifically
    print(f"Recovered J: {results_dict['J']}")


if __name__ == "__main__":
    test_j_modulated_fit_synthetic()
