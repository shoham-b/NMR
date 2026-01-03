from typing import List, Any, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d

from nmr_analysis.core.types import AnalysisResult, NMRData, ExperimentType


@dataclass
class AnalysisContext:
    """Holds data and result for a single analysis to be visualized."""

    data: NMRData
    result: AnalysisResult
    # For T2 Combined, we might have specific peak data
    peak_times: Optional[np.ndarray] = None
    peak_amps: Optional[np.ndarray] = None
    # For T1/T2, we have the list of raw traces and their extracted points
    raw_traces: Optional[List[Any]] = (
        None  # List[Tuple[NMRData, float, float, float, np.ndarray]]
    )


def generate_dashboard(contexts: List[AnalysisContext], output_path: Path):
    """
    Generates a single HTML file with a dropdown to select between different analysis results.
    """
    if not contexts:
        return

    # Create figure with 3 subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Raw Signals", "Decay (Linear)", "Decay (Log)"),
        horizontal_spacing=0.05,
    )

    # We will add traces for ALL contexts, but make them visible only when selected.
    # Each context might have multiple traces (Raw, Fit, Peaks, Residuals).
    # We need to track which traces belong to which context.

    # Generate colors from a gradient
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    num_contexts = len(contexts)
    cmap = cm.get_cmap("viridis")

    traces_per_context = []

    for i, ctx in enumerate(contexts):
        # Calculate color for this context
        ratio = i / (num_contexts - 1) if num_contexts > 1 else 0.5
        rgba = cmap(ratio)
        color_hex = mcolors.to_hex(rgba)

        start_idx = len(fig.data)
        _add_traces_for_context(fig, ctx, main_color=color_hex)
        end_idx = len(fig.data)
        traces_per_context.append(list(range(start_idx, end_idx)))

    # Create Dropdown buttons
    buttons = []
    for i, ctx in enumerate(contexts):
        # Custom visibility logic: Raw traces usually hidden by default (legendonly)
        visible_status = []
        for j, trace in enumerate(fig.data):
            if j in traces_per_context[i]:
                if trace.name and (
                    trace.name.startswith("Raw Trace")
                    or trace.name.startswith("Smoothed Trace")
                    or trace.name.startswith("All Peaks")
                ):
                    visible_status.append("legendonly")
                else:
                    visible_status.append(True)
            else:
                visible_status.append(False)

        # Layout updates
        layout_args = {"title": f"Analysis: {ctx.result.dataset_name}"}

        # Axis scaling logic
        if ctx.result.experiment_type == ExperimentType.T2_STAR:
            # Enforce start at 1 (log10(1)=0) for Log plot (yaxis3)
            layout_args["yaxis3.autorange"] = False
            layout_args["yaxis3.range"] = [0, None]
        else:
            # Auto-scale for others
            layout_args["yaxis3.autorange"] = True

        buttons.append(
            dict(
                label=f"{ctx.result.dataset_name} ({ctx.result.experiment_type.value})",
                method="update",
                args=[
                    {"visible": visible_status},
                    layout_args,
                ],
            )
        )

    # Update Layout with Dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        title=f"Analysis: {contexts[0].result.dataset_name}"
        if contexts
        else "NMR Analysis",
        template="plotly_white",
        hovermode="closest",
    )

    # Configure Axes
    # Col 1: Raw Signals
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=1, col=1)

    # Col 2: Linear Decay
    fig.update_xaxes(title_text="Delay (s)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)

    # Col 3: Log Decay
    fig.update_xaxes(title_text="Delay (s)", row=1, col=3)
    fig.update_yaxes(title_text="Amplitude (Log)", type="log", row=1, col=3)

    # Initial visibility: Show only first context, respecting legendonly
    for i, trace in enumerate(fig.data):
        if i in traces_per_context[0]:
            if trace.name and (
                trace.name.startswith("Raw Trace")
                or trace.name.startswith("Smoothed Trace")
                or trace.name.startswith("All Peaks")
            ):
                trace.visible = "legendonly"
            else:
                trace.visible = True
        else:
            trace.visible = False

    # Apply initial layout for first context
    first_ctx = contexts[0]
    if first_ctx.result.experiment_type == ExperimentType.T2_STAR:
        fig.update_layout(yaxis3_range=[0, None], yaxis3_autorange=False)
    else:
        fig.update_layout(yaxis3_autorange=True)

    # Save to HTML
    fig.write_html(str(output_path))
    print(f"Dashboard saved to {output_path}")


def _add_traces_for_context(
    fig: go.Figure, ctx: AnalysisContext, main_color: str = "red"
):
    """Helper to add traces for a specific context."""
    experiment = ctx.result.experiment_type

    # Color palette
    color_fit = main_color
    color_peaks = "green"

    # --- T1 / T2 / T2 Combined ---
    if experiment in [ExperimentType.T1, ExperimentType.T2, ExperimentType.T2_COMBINED]:
        # 1. Raw Traces (Background) -> Raw Signals Panel (Col 1)
        if ctx.raw_traces:
            for j, (rdata, rtime, ramp, rtau, all_peaks, *_) in enumerate(
                ctx.raw_traces
            ):
                fig.add_trace(
                    go.Scatter(
                        x=rdata.time,
                        y=np.abs(rdata.signal),
                        mode="lines",
                        line=dict(color="cornflowerblue", width=1),
                        opacity=0.3,
                        showlegend=True,
                        name=f"Raw Trace {j + 1} (tau={rtau:.2e})",
                        hoverinfo="skip",
                        visible=False,
                    ),
                    row=1,
                    col=1,
                )

                # Smoothed Trace (Overlay) -> Raw Signals Panel (Col 1)
                smoothed_sig = gaussian_filter1d(np.abs(rdata.signal), sigma=1.0)
                fig.add_trace(
                    go.Scatter(
                        x=rdata.time,
                        y=smoothed_sig,
                        mode="lines",
                        line=dict(color="darkblue", width=1.5),  # darker for visibility
                        opacity=0.8,
                        showlegend=True,
                        name=f"Smoothed Trace {j + 1} (sigma=1.0)",
                        hoverinfo="skip",
                        visible=False,
                    ),
                    row=1,
                    col=1,
                )

                # Extracted point on raw trace -> Raw Signals Panel (Col 1)
                fig.add_trace(
                    go.Scatter(
                        x=[rtime],
                        y=[ramp],
                        mode="markers",
                        marker=dict(color=main_color, size=6, symbol="x"),
                        showlegend=False,
                        name=f"Point {j + 1}",
                        visible=False,
                    ),
                    row=1,
                    col=1,
                )

                # All Peaks (Gray dots) -> Raw Signals Panel (Col 1)
                if len(all_peaks) > 0:
                    # Ensure all_peaks is a numpy array
                    all_peaks_array = (
                        np.array(all_peaks)
                        if not isinstance(all_peaks, np.ndarray)
                        else all_peaks
                    )

                    smoothed_for_peaks = gaussian_filter1d(
                        np.abs(rdata.signal), sigma=1.0
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=rdata.time[all_peaks_array],
                            y=smoothed_for_peaks[all_peaks_array],
                            mode="markers",
                            marker=dict(color="gray", size=5, symbol="circle"),
                            showlegend=True,
                            name=f"All Peaks {j + 1}",
                            visible=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )

        # 2. Fit/Decay Data
        # For T2 Combined, we have explicit peak times
        if experiment == ExperimentType.T2_COMBINED and ctx.peak_times is not None:
            # Points -> Linear (Col 2) AND Log (Col 3)
            for r, c in [(1, 2), (1, 3)]:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.peak_times,
                        y=ctx.peak_amps,
                        mode="markers",
                        name="Extracted Peaks",
                        marker=dict(color=color_peaks, size=8, symbol="x"),
                        visible=False,
                        showlegend=(
                            c == 3
                        ),  # Show legend once (or both? plotly merges same names)
                    ),
                    row=r,
                    col=c,
                )

            # Fit Curve -> Linear (Col 2) AND Log (Col 3)
            for r, c in [(1, 2), (1, 3)]:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.peak_times,
                        y=ctx.result.fit_curve,
                        mode="lines",
                        name=f"Fit (R²={ctx.result.r_squared:.4f})",
                        line=dict(color=color_fit, width=2, dash="dash"),
                        visible=False,
                        showlegend=(c == 3),
                    ),
                    row=r,
                    col=c,
                )

        # For T1/T2, we have delay vs signal
        elif experiment in [ExperimentType.T1, ExperimentType.T2]:
            # Decay Data Points -> Linear (Col 2) AND Log (Col 3)
            # (Note: ctx.data holds the processed decay data here)
            for r, c in [(1, 2), (1, 3)]:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.data.time,
                        y=np.abs(ctx.data.signal),
                        mode="markers",
                        name="Decay Data",
                        marker=dict(color=main_color, size=8),
                        visible=False,
                        showlegend=(c == 3),
                    ),
                    row=r,
                    col=c,
                )

            # Fit Curve -> Linear (Col 2) AND Log (Col 3)
            sorted_pairs = sorted(zip(ctx.data.time, ctx.result.fit_curve))
            sx, sy = zip(*sorted_pairs)
            for r, c in [(1, 2), (1, 3)]:
                fig.add_trace(
                    go.Scatter(
                        x=sx,
                        y=sy,
                        mode="lines",
                        name=f"Fit (R²={ctx.result.r_squared:.4f})",
                        line=dict(color=color_fit, width=2),
                        visible=False,
                        showlegend=(c == 3),
                    ),
                    row=r,
                    col=c,
                )

        # Raw Data for T2 Combined (it has raw echo train) -> Raw Signals (Col 1) Only
        # (Checking context structure, T2 Combined usually puts raw train in ctx.data)
        if (
            experiment == ExperimentType.T2_COMBINED
            and ctx.data
            and ctx.data.signal is not None
        ):
            # This is the full echo train
            fig.add_trace(
                go.Scatter(
                    x=ctx.data.time,
                    y=np.abs(ctx.data.signal),
                    mode="lines",
                    name="Raw Echo Train",
                    line=dict(color=main_color, width=1),
                    visible=False,
                ),
                row=1,
                col=1,
            )

            # Overlay Fit Envelope on Raw Echo Train
            if ctx.peak_times is not None and ctx.result.fit_curve is not None:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.peak_times,
                        y=ctx.result.fit_curve,
                        mode="lines",
                        name=f"Fit Envelope (R²={ctx.result.r_squared:.4f})",
                        line=dict(color=color_fit, width=3, dash="dash"),
                        visible=False,
                    ),
                    row=1,
                    col=1,
                )

    # --- T2* ---
    elif experiment == ExperimentType.T2_STAR:
        # Filter > 1 for Log plot
        y_data = np.abs(ctx.data.signal)
        mask_log = y_data > 1

        # 1. Raw Data -> Cols 1, 2 (Linear) and 3 (Log)
        # Linear (Col 1, 2)
        for c in [1, 2]:
            fig.add_trace(
                go.Scatter(
                    x=ctx.data.time,
                    y=y_data,
                    mode="lines",
                    name="Raw Data",
                    line=dict(color=main_color, width=1),
                    visible=False,
                    showlegend=(c == 2),
                ),
                row=1,
                col=c,
            )

        # Log (Col 3) - Filtered
        fig.add_trace(
            go.Scatter(
                x=ctx.data.time[mask_log],
                y=y_data[mask_log],
                mode="lines",
                name="Raw Data",
                line=dict(color=main_color, width=1),
                visible=False,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # 2. Fit Curve -> Cols 1, 2 (Linear) and 3 (Log)
        fit_y = ctx.result.fit_curve
        if fit_y is not None:
            # Linear (Col 1, 2)
            for c in [1, 2]:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.data.time,
                        y=fit_y,
                        mode="lines",
                        name=f"Fit (R²={ctx.result.r_squared:.4f})",
                        line=dict(color=color_fit, width=2),
                        visible=False,
                        showlegend=(c == 2),
                    ),
                    row=1,
                    col=c,
                )

            # Log (Col 3) - Filtered
            fit_mask = fit_y > 1

            # Format label with error if available
            label_fit = f"Fit (R²={ctx.result.r_squared:.4f})"
            if "T2_star" in ctx.result.params:
                val = ctx.result.params["T2_star"]
                err = ctx.result.param_errors.get("T2_star", 0.0)
                label_fit = (
                    f"Fit (T2*={val:.4f}±{err:.4f}, R²={ctx.result.r_squared:.4f})"
                )

            fig.add_trace(
                go.Scatter(
                    x=ctx.data.time[fit_mask],
                    y=fit_y[fit_mask],
                    mode="lines",
                    name=label_fit,
                    line=dict(color=color_fit, width=2),
                    visible=False,
                    showlegend=(c == 3),
                ),
                row=1,
                col=3,
            )
