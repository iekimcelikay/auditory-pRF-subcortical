import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from pathlib import Path


def hz_formatter(x, pos):
    if x >= 1000:
        return f'{x/1000:.1f}k'
    else:
        return f'{int(x)}'

def plot_spectrogram_forCFS(cf_list, tone_freqs, response_matrix, db_level, save_path=None):
    """Plot spectrogram showing CF responses to different tone frequencies.

    Args:
        cf_list: Array of CF values (Hz)
        tone_freqs: Array of tone frequencies (Hz)
        response_matrix: 2D array of shape (num_cf, num_tones) with firing rates
        db_level: dB level of stimuli (for title)
        save_path: Optional path to save figure

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Transpose matrix to have CFs on x-axis
    # Original: (num_cf, num_tones), Transposed: (num_tones, num_cf)
    response_matrix_T = response_matrix.T

    # Create heatmap with CFs on x-axis
    im = ax.imshow(response_matrix_T,
                   aspect='auto',
                   origin='lower',
                   cmap='viridis',
                   interpolation='nearest',
                   extent=[cf_list[0], cf_list[-1], tone_freqs[0], tone_freqs[-1]])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Firing Rate (spikes/s)')

    # Labels and title
    ax.set_xlabel('CF (Hz)', fontsize=12)
    ax.set_ylabel('Tone Frequency (Hz)', fontsize=12)
    ax.set_title(f'Cochlear Spectrogram - Population Response at {db_level} dB SPL',
                 fontsize=14, fontweight='bold')

    # Set log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set ticks to show all values
    ax.set_xticks(cf_list)
    ax.set_yticks(tone_freqs)

    # Format tick labels to show actual Hz values
    ax.xaxis.set_major_formatter(FuncFormatter(hz_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(hz_formatter))

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax

def plot_psth_timecourses(time_axis, population_psth, cf_list, identifier="", save_path=None):
    """Plot PSTH time courses for all cochlear channels.

    Args:
        time_axis: Array of time points (seconds)
        population_psth: 2D array of shape (num_cf, n_bins) - population response over time
        cf_list: Array of CF values (Hz)
        identifier: Optional identifier for the plot title
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    num_cf = len(cf_list)

    # Create subplot grid
    n_cols = min(3, num_cf)
    n_rows = int(np.ceil(num_cf / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))

    # Handle single subplot case
    if num_cf == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot time course for each CF channel
    for i_cf, cf in enumerate(cf_list):
        ax = axes[i_cf]

        # Get population response time course for this CF
        timecourse = population_psth[i_cf, :]

        # Plot time course
        ax.plot(time_axis, timecourse, linewidth=1.5, color='steelblue')

        # Labels and formatting
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Firing Rate (sp/s)', fontsize=10)
        ax.set_title(f'CF = {cf:.0f} Hz', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=9)

    # Hide unused subplots
    for i in range(num_cf, len(axes)):
        axes[i].axis('off')

    # Overall title
    title = f'PSTH Time Courses - Population Response'
    if identifier:
        title += f' ({identifier})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_timecourse_per_cf(
    time_axis,
    population_psth,
    cf_list,
    identifier="",
    save_dir=None,
    dpi=300,
    figsize=(18, 4),
    tone_markers=None,
):
    """Plot the firing-rate time course for each CF in a separate high-quality figure.

    Each figure spans the full stimulus duration with the x-axis in milliseconds.
    Tick marks every 10 ms; labels every 50 ms.  Optional red vertical lines mark
    tone onsets (solid) and offsets (dashed) for inspection of silence periods.

    Args:
        time_axis (array-like): 1-D array of time points in **seconds**.
        population_psth (ndarray): 2-D array of shape (num_cf, n_bins) – firing rates (sp/s).
        cf_list (array-like): 1-D array of CF values in Hz.
        identifier (str): String identifier used in titles and file names.
        save_dir (Path | str | None): Directory where figures are saved.
            Created automatically if it does not exist. Pass None to skip saving.
        dpi (int): Resolution in dots per inch (default 300 for high quality).
        figsize (tuple): Figure size (width, height) in inches.
            Default (18, 4) gives a wide, detailed view of the time course.
        tone_markers (tuple | None): If provided, a tuple ``(tone_dur_ms, isi_ms)``
            used to compute and draw red vertical lines at every tone onset (solid)
            and offset (dashed) across the full time axis.

    Returns:
        list[Figure]: One Matplotlib Figure per CF.
    """
    time_ms = np.asarray(time_axis) * 1000.0  # seconds → milliseconds
    total_ms = time_ms[-1]

    # Pre-compute tone onset / offset times (ms) if timing info is provided
    if tone_markers is not None:
        tone_dur_ms, isi_ms = tone_markers
        period_ms = tone_dur_ms + isi_ms
        onsets_ms  = np.arange(0, total_ms, period_ms)
        offsets_ms = onsets_ms + tone_dur_ms
        # Drop offsets that exceed the total duration
        offsets_ms = offsets_ms[offsets_ms <= total_ms]
    else:
        onsets_ms = offsets_ms = np.array([])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    for i_cf, cf in enumerate(cf_list):
        fig, ax = plt.subplots(figsize=figsize)

        timecourse = population_psth[i_cf, :]
        ax.plot(time_ms, timecourse, linewidth=0.8, color="steelblue", zorder=2)

        # ── tone onset / offset markers ─────────────────────────────────────────
        for k, t_on in enumerate(onsets_ms):
            ax.axvline(t_on,  color="red", linewidth=0.8, linestyle="-",  alpha=0.7, zorder=3,
                       label="tone onset"  if k == 0 else "_nolegend_")
        for k, t_off in enumerate(offsets_ms):
            ax.axvline(t_off, color="red", linewidth=0.8, linestyle="--", alpha=0.5, zorder=3,
                       label="tone offset" if k == 0 else "_nolegend_")
        if len(onsets_ms) or len(offsets_ms):
            ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

        # ── x-axis: tick marks every 10 ms, labels every 50 ms ─────────────────
        ax.set_xlim(time_ms[0], total_ms)
        ax.xaxis.set_major_locator(MultipleLocator(50))   # labeled every 50 ms
        ax.xaxis.set_minor_locator(MultipleLocator(10))   # tick mark every 10 ms
        ax.tick_params(axis="x", which="major", length=5, labelsize=7.5)
        ax.tick_params(axis="x", which="minor", length=3, labelsize=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # ── y-axis ──────────────────────────────────────────────────────────────
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=9)

        # ── labels & title ──────────────────────────────────────────────────────
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Firing Rate (sp/s)", fontsize=11)

        cf_label = f"{cf / 1000:.2f} kHz" if cf >= 1000 else f"{cf:.0f} Hz"
        title = f"CF = {cf_label}"
        if identifier:
            title = f"{identifier}  |  {title}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        # ── grid ────────────────────────────────────────────────────────────────
        ax.grid(True, which="major", alpha=0.3, linestyle="--", linewidth=0.5)  # every 50 ms
        ax.grid(True, which="minor", alpha=0.1, linestyle=":",  linewidth=0.3)  # every 10 ms

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        # ── save ─────────────────────────────────────────────────────────────────
        if save_dir is not None:
            cf_str = f"{cf:.0f}Hz"
            fname = f"{identifier}_CF_{cf_str}.png" if identifier else f"CF_{cf_str}.png"
            fpath = save_dir / fname
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {fpath.name}")

        figs.append(fig)

    return figs


def plot_tuning_curves(cf_list, tone_freqs, response_matrix, db_level, save_path=None):
    """Plot tuning curves for each CF channel in subplots.

    Args:
        cf_list: Array of CF values (Hz)
        tone_freqs: Array of tone frequencies (Hz)
        response_matrix: 2D array of shape (num_cf, num_tones) with firing rates
        db_level: dB level of stimuli (for suptitle)
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    num_cf = len(cf_list)

    # Create subplot grid (8 rows x 5 columns for 40 CFs)
    n_rows = 8
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    axes = axes.flatten()

    # Plot tuning curve for each CF
    for i_cf, cf in enumerate(cf_list):
        ax = axes[i_cf]

        # Get firing rates for this CF across all tone frequencies
        tuning_curve = response_matrix[i_cf, :]

        # Plot tuning curve
        ax.plot(tone_freqs, tuning_curve, 'o-', linewidth=2, markersize=4, color='steelblue')

        # Mark the CF on the plot
        ax.axvline(cf, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label=f'CF={cf:.0f} Hz')

        # Labels and formatting
        ax.set_xscale('log')
        ax.set_xlabel('Tone Freq (Hz)', fontsize=8)
        ax.set_ylabel('Rate (sp/s)', fontsize=8)
        ax.set_title(f'CF = {cf:.0f} Hz', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)

        # Format x-axis tick labels
        ax.xaxis.set_major_formatter(FuncFormatter(hz_formatter))

    # Overall title
    fig.suptitle(f'Tuning Curves for All CF Channels at {db_level} dB SPL',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes