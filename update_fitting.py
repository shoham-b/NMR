import os

file_path = r"src/nmr_analysis/analysis/fitting.py"

new_method = """    @staticmethod
    def fit_spectrum(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        method: str = "magnitude_lorentzian",
        min_prominence: Optional[float] = None,
    ) -> AnalysisResult:
        \"\"\"
        Fit the frequency spectrum with multiple Lorentzians.
        1. Find peaks in magnitude spectrum.
        2. Fit sum of Magnitude Lorentzians (if spectrum is magnitude).
        3. Extract T2* from gamma. T2* = 1 / (2 * pi * gamma).
        \"\"\"
        from scipy.signal import find_peaks
        from nmr_analysis.analysis.models import multi_magnitude_lorentzian

        mag_spec = np.abs(spectrum)

        # 1. PEAK FINDING
        if min_prominence is None:
            min_prominence = 0.05 * np.max(mag_spec)

        peak_indices, properties = find_peaks(
            mag_spec, prominence=min_prominence, distance=5
        )

        if len(peak_indices) == 0:
            peak_indices = [np.argmax(mag_spec)]

        n_peaks = len(peak_indices)
        
        # 2. INITIAL GUESS
        current_offset = np.median(mag_spec)
        p0 = [current_offset]
        
        # Bounds: [offset, A1, f0_1, gamma1, ...]
        bounds_min = [-np.inf]
        bounds_max = [np.inf]

        # Sort peaks by amplitude
        sorted_indices = sorted(peak_indices, key=lambda i: mag_spec[i], reverse=True)
        if len(sorted_indices) > 5:
             sorted_indices = sorted_indices[:5]
             n_peaks = 5
        
        # Sort by frequency
        sorted_indices = sorted(sorted_indices)

        for idx in sorted_indices:
            amp = mag_spec[idx]
            f0 = freqs[idx]
            df = freqs[1] - freqs[0]
            gamma_guess = 5.0 * df # initial width guess
            
            # Initial Guess for A
            # Model: MagnitudeLorentzian Height = A / gamma + offset
            # => A = (Height - offset) * gamma
            height = amp - current_offset
            A_guess = max(height * gamma_guess, 0.0)
            
            p0.extend([A_guess, f0, gamma_guess])
            
            bounds_min.extend([0, freqs[0], 0])
            bounds_max.extend([np.inf, freqs[-1], np.inf])

        # 3. FITTING
        try:
            def fit_func(f, *params):
                 return multi_magnitude_lorentzian(f, np.array(params), n_peaks)

            popt, pcov = curve_fit(
                fit_func, 
                freqs, 
                mag_spec, 
                p0=p0,
                bounds=(bounds_min, bounds_max)
            )

            # 4. EXTRACT RESULTS
            offset_fit = popt[0]
            peaks_found = []
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
            
            fit_curve = fit_func(freqs, *popt)
            residuals = mag_spec - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mag_spec - np.mean(mag_spec))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            for i in range(n_peaks):
                base_idx = 1 + i*3
                A = popt[base_idx]
                f0 = popt[base_idx+1]
                gamma = popt[base_idx+2]
                
                # T2* conversion
                # Gamma is parameter in sqrt((f-f0)^2 + gamma^2)
                # It corresponds to 1 / (2*pi*T2)
                t2_star = 0.0
                if gamma > 1e-9:
                    t2_star = 1.0 / (2.0 * np.pi * gamma)

                peaks_found.append({
                    "amplitude": A,
                    "f0": f0,
                    "gamma": gamma,
                    "t2_star": t2_star,
                    "amplitude_error": perr[base_idx],
                    "f0_error": perr[base_idx+1],
                    "gamma_error": perr[base_idx+2]
                })

            # Sort peaks by f0 for consistent reporting
            peaks_found.sort(key=lambda x: x["f0"])

            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR, 
                dataset_name="Spectral Analysis",
                params={
                    "offset": offset_fit,
                    "peaks": peaks_found,
                    "n_peaks": n_peaks
                },
                fit_curve=fit_curve,
                residuals=residuals,
                r_squared=r2,
                param_errors={}, 
                metadata={
                    "freqs": freqs,
                    "spectrum_magnitude": mag_spec
                }
            )

        except Exception as e:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Spectral Analysis (Fit Failed)",
                params={"peaks": [], "error": str(e)},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
                metadata={}
            )
"""

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Find start of fit_spectrum
marker = "    @staticmethod\n    def fit_spectrum("
idx = content.find(marker)

if idx == -1:
    print("Could not find start of fit_spectrum")
    exit(1)

# Keep content before marker
new_content = content[:idx] + new_method

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully updated fitting.py")
