import numpy as np
import sys

print("Starting debug...")

try:
    from nmr_analysis.analysis.models import multi_lorentzian

    print("Imported multi_lorentzian")
except Exception as e:
    print(f"Failed to import multi_lorentzian: {e}")

try:
    from nmr_analysis.analysis.processing import compute_spectrum

    print("Imported compute_spectrum")
except Exception as e:
    print(f"Failed to import compute_spectrum: {e}")

try:
    from nmr_analysis.analysis.fitting import Fitter

    print("Imported Fitter")
except Exception as e:
    print(f"Failed to import Fitter: {e}")

# Try running multi_lorentzian (Numba check)
try:
    f = np.linspace(0, 100, 100)
    params = np.array([0.0, 1.0, 50.0, 5.0])  # offset, A, f0, gamma
    y = multi_lorentzian(f, params, 1)
    print("Ran multi_lorentzian success")
except Exception as e:
    print(f"Failed to run multi_lorentzian: {e}")

# Try fit logic
try:
    res = Fitter.fit_spectrum(f, y)
    print("Ran fit_spectrum success")
    print(res.params)
except Exception as e:
    print(f"Failed to run fit_spectrum: {e}")

print("Debug finished")
