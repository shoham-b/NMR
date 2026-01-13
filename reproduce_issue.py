import numpy as np
import matplotlib.pyplot as plt
from nmr_analysis.analysis.fitting import Fitter


def generate_synthetic_data(t, M0, T2, J, offset, noise_level=0.0):
    # Model: | M0 * exp(-t/T2) * cos(pi * J * t) | + offset
    signal = np.abs(M0 * np.exp(-t / T2) * np.cos(np.pi * J * t)) + offset
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, size=len(t))
    return np.abs(signal)  # Magnitude is always positive


def test_fitting():
    # Parameters matching rough observations from user image
    # T2 ~ 0.1s, J ~ 7Hz? User got J=3.68Hz.
    # Delays: 0.1 to 0.8s

    true_M0 = 100.0
    true_T2 = 0.2  # Let's try 0.2s
    true_J = 7.0  # Ethanol approx
    true_offset = 2.0

    # Sparse delays like user seems to have
    delays = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    print(f"True Params: M0={true_M0}, T2={true_T2}, J={true_J}, offset={true_offset}")

    amplitudes = generate_synthetic_data(
        delays, true_M0, true_T2, true_J, true_offset, noise_level=2.0
    )

    print("Delays:", delays)
    print("Amplitudes:", amplitudes)

    # Run Fit
    result, fit_curve, residuals, r2, errors = Fitter.fit_modulated_t2(
        delays, amplitudes, guess_J=7.0
    )

    with open("reproduce_output.txt", "w") as f:
        f.write(
            f"True Params: M0={true_M0}, T2={true_T2}, J={true_J}, offset={true_offset}\n"
        )
        f.write(f"Delays: {delays}\n")
        f.write(f"Amplitudes: {amplitudes}\n")
        f.write("\nFit Results:\n")
        f.write(str(result) + "\n")
        f.write(f"R2: {r2}\n")
        f.write(f"Errors: {errors}\n")

    print("Saved reproduce_output.txt")

    # Plot
    t_dense = np.linspace(0, 1.0, 1000)
    y_dense_true = generate_synthetic_data(
        t_dense, true_M0, true_T2, true_J, true_offset
    )

    if result:
        from nmr_analysis.analysis.models import j_modulated_t2

        y_dense_fit = j_modulated_t2(
            t_dense, result["M0"], result["T2"], result["J"], result["offset"]
        )
    else:
        y_dense_fit = np.zeros_like(t_dense)

    plt.figure()
    plt.plot(delays, amplitudes, "bo", label="Data")
    plt.plot(t_dense, y_dense_true, "g--", label="True", alpha=0.5)
    plt.plot(t_dense, y_dense_fit, "r-", label="Fit")
    plt.legend()
    plt.title(f"Fit Test: J_in={true_J}, J_out={result.get('J', 'N/A'):.2f}")
    plt.savefig("reproduce_fit.png")
    print("Saved reproduce_fit.png")


if __name__ == "__main__":
    test_fitting()
