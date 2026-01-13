import os

file_path = r"src/nmr_analysis/cli/commands.py"

spectrum_block = """    elif experiment == ExperimentType.SPECTRUM:
        # Spectrum Analysis
        target_files = []
        if path.is_dir():
             target_files = list(path.glob("*"))
             # Filter common extensions
             target_files = [f for f in target_files if f.suffix.lower() in [".h5", ".hdf5", ".csv"]]
        else:
             target_files = [path]

        console.print(f"Found {len(target_files)} files for Spectral analysis.")

        for target_file in target_files:
            try:
                console.print(f"Loading {target_file.name}...")
                loader = get_loader(target_file, channel=channel)
                data = loader.load(target_file)
                
                console.print("Computing Spectrum...")
                freqs, spect = compute_spectrum(data)
                
                console.print("Fitting Spectrum (Magnitude)...")
                result = Fitter.fit_spectrum(freqs, spect)
                
                # Add dataset name
                if len(target_files) > 1:
                     result.dataset_name = f"{result.dataset_name} ({target_file.stem})"

                print_result(result)
                
                if plot:
                    filepath = None
                    if save_path:
                        out_dir = save_path if save_path.is_dir() else save_path.parent
                        fname = f"{prefix}_{target_file.stem}_spectrum_fit.png" if prefix else f"{target_file.stem}_spectrum_fit.png"
                        filepath = out_dir / fname
                        console.print(f"Saving plot to {filepath}")
                    
                    plot_spectrum_fit(
                        freqs, 
                        np.abs(spect), 
                        result, 
                        filepath=filepath
                    )
                
                results.append(AnalysisContext(data=data, result=result))

            except Exception as e:
                console.print(f"[red]Failed to analyze {target_file.name}: {e}[/red]")
        
        return results

"""

plot_func = """
def plot_spectrum_fit(freqs, mag_data, result, filepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, mag_data, label="Data (Magnitude)", color="black", alpha=0.7)
    if len(result.fit_curve) > 0:
        ax.plot(freqs, result.fit_curve, label="Fit (Mag Lorentzian)", color="red", linestyle="--")
    
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

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Insert Spectrum block before Diffusion
marker = "    elif experiment == ExperimentType.DIFFUSION:"
idx = content.find(marker)

if idx == -1:
    print("Could not find marker for Diffusion block")
    exit(1)

new_content = content[:idx] + spectrum_block + content[idx:]
new_content += plot_func

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully updated commands.py")
