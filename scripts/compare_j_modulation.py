import numpy as np
import matplotlib.pyplot as plt
import os


def generate_comparison():
    # Parameters
    M0 = 100
    T2 = 0.2  # 200ms
    J = 7.0  # 7 Hz

    t = np.linspace(0, 1.0, 1000)

    # Models
    # 1. Standard Exponential
    signal_std = M0 * np.exp(-t / T2)

    # 2. J-Modulated
    signal_j = np.abs(M0 * np.exp(-t / T2) * np.cos(np.pi * J * t))

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Linear Scale
    ax1.plot(t, signal_std, "b--", label="Standard Decay (No J)", alpha=0.6)
    ax1.plot(t, signal_j, "r-", label=f"J-Modulated (J={J}Hz)")
    ax1.set_title("Linear Scale")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log Scale
    # Add small offset to avoid log(0) for J-mod zero crossings perfectly
    safe_signal_j = signal_j.copy()
    safe_signal_j[safe_signal_j < 1e-3] = 1e-3

    ax2.semilogy(t, signal_std, "b--", label="Standard Decay")
    ax2.semilogy(t, safe_signal_j, "r-", label="J-Modulated")
    ax2.set_title("Log Scale")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Log Amplitude")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    output_path = "j_modulation_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Generated plot at: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    generate_comparison()
