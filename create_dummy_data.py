import os
import csv
import numpy as np
from pathlib import Path

# Parameters
GAMMA = 2.675e8
G = 1.0  # T/m
D_TRUE = 2.0e-9  # m^2/s
T2_INTRINSIC = 2.0  # s
R2_INTRINSIC = 1.0 / T2_INTRINSIC

BASE_DIR = Path("dummy_data/Water")
T2_DIR = BASE_DIR / "t2"
T2M_DIR = BASE_DIR / "t2multiple"


def create_csv(filepath, t, s, metadata={}):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        # Row 0, 1: Metadata
        keys = list(metadata.keys())
        values = list(metadata.values())
        for i in range(max(len(keys), 2)):  # Fill at least 2 rows
            k = keys[i] if i < len(keys) else ""
            v = values[i] if i < len(values) else ""
            writer.writerow([k, v, "", "", "", "", ""])

        # Row 2: Headers (Skipped by loader logic slice(2) effectively skips 0,1.
        # Wait, loader.py does df.slice(2).
        # slice(2) removes rows 0 and 1. So row 2 becomes row 0 of data frame.
        # But the loader expects data to be in the data frame.
        # It casts columns.
        # If headers are at row 2 of original file, they will be row 0 of dataframe.
        # Polars read_csv with has_header=False.
        # So row 2 is data? No.
        # "The first two rows of data columns (3, 4, 5) are headers and should be skipped for data."
        # This comment in loader.py is slightly confusing.
        # Let's assume slice(2) skips metadata.
        # Then headers might be at start of slice?
        # But loader just casts to Float64. If headers are strings, cast will fail (become null).
        # Loader filters nulls.
        # So headers at row 2 are fine, they will be filtered out.

        writer.writerow(["Header", "Val", "", "Time", "Ch1", "Ch2", ""])

        # Data
        for ti, si in zip(t, s):
            writer.writerow(["", "", "", ti, 0, si, ""])  # Signal in Ch2


def generate_decay(t_axis, T2, M0=1.0):
    return M0 * np.exp(-t_axis / T2)


def main():
    print(f"Generating dummy data in {BASE_DIR}...")

    # 1. T2 Combined (Intrinsic T2)
    # Echo Train
    t_echo = np.linspace(0.001, 5.0, 50)  # 5 seconds decay
    s_echo = generate_decay(t_echo, T2_INTRINSIC, M0=1000.0)

    # Add fake raw points around peaks
    # Loader extracts peak train? No, commands.py extracts peaks.
    # extract_echo_train does smoothing and peak finding.
    # We should generate a signal that LOOKS like an echo train (peaks)
    # or just simple decay if extract_echo_train works on envelopes?
    # commands.py: `peak_times, peak_amps = extract_echo_train(data)`
    # extract_echo_train relies on finding peaks.
    # So we need to simulate actual echoes? That's hard.
    # Or just a dataset where there ARE peaks.
    # Let's create a signal that is 0 everywhere except at echo times.
    # And add some width to peaks.

    time_full = np.linspace(0, 5.0, 5000)
    signal_full = np.zeros_like(time_full)

    # Echoes every 0.1s
    te = 0.1
    echo_times = np.arange(te, 5.0, te)
    for et in echo_times:
        # Gaussian peak at et
        width = 0.005
        amp = 1000.0 * np.exp(-et / T2_INTRINSIC)
        signal_full += amp * np.exp(-((time_full - et) ** 2) / (2 * width**2))

    create_csv(
        T2M_DIR / "combined.csv", time_full, signal_full, {"Note": "T2 Combined"}
    )

    # 2. Diffusion (Variable Tau)
    # We need multiple files with different tau.
    # Each file is a T2 decay (Spin Echo).
    # But only one peak is needed? Or decay?
    # commands.py: `peak_times, peak_amps = extract_echo_train...`
    # Wait, for T2 (Diffusion), commands.py does `extract_echo_train` too (lines 472).
    # And picks peaks.
    # So each diffusion file is ALSO an echo train (CPMG) or just a single echo?
    # `taus.append(tau_val)`.
    # `rates.append(r2_obs)`.
    # `r2_obs = 1.0 / t2_obs`.
    # `t2_obs` is fitted from the echo train of THAT trace.
    # So for each tau, we have a CPMG train.
    # The DECAY RATE of that train is R2_obs.
    # R2_obs varies with tau.

    taus = [0.001, 0.002, 0.003, 0.004, 0.005]

    for tau in taus:
        # Calculate expected R2_obs
        # R2_obs = R2_int + K * tau^2
        K = (1 / 3) * D_TRUE * (GAMMA**2) * (G**2)
        R2_obs = R2_INTRINSIC + K * (tau**2)
        T2_obs = 1.0 / R2_obs

        # Generate Echo Train for this T2_obs
        # Echo spacing? Assuming fixed TE for the CPMG part?
        # The 'tau' in filename is the diffusion block delay?
        # Let's just create peaks decaying with T2_obs.

        signal_tau = np.zeros_like(time_full)
        for et in echo_times:
            width = 0.005
            amp = 1000.0 * np.exp(-et / T2_obs)
            signal_tau += amp * np.exp(-((time_full - et) ** 2) / (2 * width**2))

        filename = f"diffusion_tau_{tau:.6f}.csv"
        create_csv(T2_DIR / filename, time_full, signal_tau, {"Tau": tau})


if __name__ == "__main__":
    main()
