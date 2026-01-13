import numpy as np
from nmr_analysis.analysis.models import multi_multiplet_lorentzian


def test_model():
    f = np.linspace(0, 500, 1000)
    # params: offset=0, A=1, center=100, J=10, gamma=2
    params = np.array([0.0, 1.0, 100.0, 10.0, 2.0])
    multiplicities = np.array([3], dtype=np.int32)
    n = 1

    print("Running model...")
    y = multi_multiplet_lorentzian(f, params, multiplicities, n)
    print("Model ran successfully.")
    print("Y sum:", np.sum(y))


if __name__ == "__main__":
    test_model()
