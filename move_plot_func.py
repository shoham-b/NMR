import re

file_path = r"src/nmr_analysis/cli/commands.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Regex to find the function at the end
# It starts with def plot_spectrum_fit... and ends before if __name__... or EOF
# But simpler: just find the string and cut it out.
func_def = """def plot_spectrum_fit(freqs, mag_data, result, filepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, mag_data, label="Data (Magnitude)", color="black", alpha=0.7)
    if len(result.fit_curve) > 0:
        ax.plot(
            freqs,
            result.fit_curve,
            label="Fit (Mag Lorentzian)",
            color="red",
            linestyle="--"
        )

    # Mark peaks
    if "peaks" in result.params:
        for p in result.params["peaks"]:
            f0 = p["f0"]
            ax.axvline(f0, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{result.dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()"""

# The content in file might have slight whitespace differences (black formatting).
# Let's try to identify it by start and end lines.
start_marker = "def plot_spectrum_fit(freqs, mag_data, result, filepath=None):"
end_marker = "plt.show()"

start_idx = content.rfind(start_marker)
if start_idx == -1:
    print("Could not find plot_spectrum_fit at the end")
    # It might be because of formatting.
    # We will ignore removal if not found specifically at end, but we MUST insert it.
    pass
else:
    # Find the end of the function
    # It should be the last plt.show()
    # Or just cut from start_idx to the next if __name__ or EOF
    next_block = content.find('if __name__ == "__main__":', start_idx)
    if next_block != -1:
        # Cut strict
        old_func = content[start_idx:next_block]
        content = content[:start_idx] + content[next_block:]
    else:
        # Check if it goes to EOF
        content = content[:start_idx]

# Now insert it before print_result
insert_marker = "def print_result(result: AnalysisResult):"
insert_idx = content.find(insert_marker)

if insert_idx == -1:
    print("Could not find print_result")
    exit(1)

# Function to insert
valid_func = """
def plot_spectrum_fit(freqs, mag_data, result, filepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, mag_data, label="Data (Magnitude)", color="black", alpha=0.7)
    if len(result.fit_curve) > 0:
        ax.plot(
            freqs,
            result.fit_curve,
            label="Fit (Mag Lorentzian)",
            color="red",
            linestyle="--",
        )

    # Mark peaks
    if "peaks" in result.params:
        for p in result.params["peaks"]:
            f0 = p["f0"]
            ax.axvline(f0, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{result.dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

"""

new_content = content[:insert_idx] + valid_func + content[insert_idx:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully moved plot_spectrum_fit")
