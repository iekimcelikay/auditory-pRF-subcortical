"""
save_tone_clouds_gaussian_prf.py

Generates pure-tone cloud sequences whose frequencies are drawn from a
Gaussian filterbank (ERB-spaced, bell-curve counts per bin).  Frequencies
are fixed across runs via a master list (same seed → same stimuli).

For each combination of Gaussian band × duration condition, one 20-second
sequence is saved as a WAV file.

Output filename convention:
    tonecloud{g+1:02d}_fc{center_hz}hz_dur{D}ms_isi{I}ms_numtones{N}.wav

Usage
-----
    python stimuli/save_tone_clouds_gaussian_prf.py
"""

import sys
from pathlib import Path
from datetime import datetime

root        = str(Path(__file__).resolve().parents[1])
stimuli_dir = str(Path(__file__).resolve().parent)
for p in (root, stimuli_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from soundgen import SoundGen
from save_sound import save_sequence_as_wav
from sample_tone_cloud_freqs_gaussian import (
    fit_gaussian_filterbank,
    build_master_list,
    sample_all_conditions,
    calculate_num_tones,
    TONE_ON_MS, ISI_MS, TOTAL_DURATION_S, SAMPLE_RATE,
)

# ==============================================================================
# CONFIG  — edit everything here
# ==============================================================================

# -- Filterbank ----------------------------------------------------------------
FREQ_MIN_HZ  = 450.0
FREQ_MAX_HZ  = 1600.0
N_GAUSSIANS  = 3
K_SIGMA      = 2.0        # boundaries at ±k·σ from each Gaussian centre

# -- Master list ---------------------------------------------------------------
N_BINS       = 10         # ERB bins per Gaussian (fewer = coarser bell curve)
N_MASTER     = 1000       # total frequencies in master pool per Gaussian
MASTER_SEED  = 0          # fixed seed → same frequencies every run

# -- Tone parameters -----------------------------------------------------------
NUM_HARMONICS   = 1       # 1 = pure sine tone
HARMONIC_FACTOR = 1.0
DBSPL           = 65
TAU_RAMP        = 0.005   # s — onset/offset ramp

# -- Stimulus timing (shared with other scripts) -------------------------------
# TONE_ON_MS and ISI_MS are imported from sample_tone_cloud_freqs_gaussian

# -- Output --------------------------------------------------------------------
STEREO       = True
WAV_SUBTYPE  = "FLOAT"
BASE_OUT_DIR = Path(__file__).parent / "produced"
RUN_PREFIX   = "toneclouds_gaussianprf"


# ==============================================================================
# HELPERS
# ==============================================================================

def _cond_id(
    gaussian_idx: int,
    center_hz:    float,
    dur_ms:       float,
    isi_ms:       float,
    num_tones:    int,
) -> str:
    return (
        f"tonecloud{gaussian_idx + 1:02d}"
        f"_fc{int(round(center_hz))}hz"
        f"_dur{int(round(dur_ms))}ms"
        f"_isi{int(round(isi_ms))}ms"
        f"_numtones{num_tones}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir   = BASE_OUT_DIR / f"{RUN_PREFIX}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fb      = fit_gaussian_filterbank(FREQ_MIN_HZ, FREQ_MAX_HZ, N_GAUSSIANS, K_SIGMA)
    masters = [
        build_master_list(fb, gaussian_idx=i, n_bins=N_BINS, n_master=N_MASTER, seed=MASTER_SEED + i)
        for i in range(fb.n)
    ]
    results = sample_all_conditions(
        fb,
        tone_on_ms=TONE_ON_MS,
        isi_ms=ISI_MS,
        total_dur_s=TOTAL_DURATION_S,
        sample_rate=SAMPLE_RATE,
        masters=masters,
    )

    soundgen     = SoundGen(SAMPLE_RATE, TAU_RAMP)
    total_files  = len(results)
    center_hz    = fb.means_hz

    print(f"Saving {total_files} tone-cloud files to: {out_dir}")
    print(f"  Band centres : {[f'{c:.1f}' for c in center_hz]} Hz")
    print(f"  Durations    : {TONE_ON_MS} ms")
    print()

    file_count = 0
    for dur_ms, isi_ms_val in zip(TONE_ON_MS, ISI_MS):
        tone_s = dur_ms     / 1000.0
        isi_s  = isi_ms_val / 1000.0
        num_tones, _, _ = calculate_num_tones(tone_s, isi_s, TOTAL_DURATION_S, SAMPLE_RATE)

        for g_idx in range(fb.n):
            freqs_hz = results[(g_idx, dur_ms)]

            sequence = soundgen.generate_sequence_from_freq_array(
                frequencies    = freqs_hz,
                num_harmonics  = NUM_HARMONICS,
                tone_duration  = tone_s,
                harmonic_factor= HARMONIC_FACTOR,
                dbspl          = DBSPL,
                isi            = isi_s,
                total_duration = TOTAL_DURATION_S,
                stereo         = STEREO,
            )

            cond_id  = _cond_id(g_idx, float(center_hz[g_idx]), dur_ms, isi_ms_val, num_tones)
            filename = f"{cond_id}.wav"
            save_sequence_as_wav(
                sequence, SAMPLE_RATE, str(out_dir / filename), subtype=WAV_SUBTYPE,
            )

            file_count += 1
            print(f"  [{file_count:>3}/{total_files}] {filename}")

    # Silence WAV — null trial
    n_silence = int(SAMPLE_RATE * TOTAL_DURATION_S)
    silence   = np.zeros((n_silence, 2) if STEREO else (n_silence,), dtype=np.float32)
    save_sequence_as_wav(
        silence, SAMPLE_RATE, str(out_dir / "tonecloud00_dur0ms_isi0ms.wav"),
        subtype=WAV_SUBTYPE,
    )
    print(f"  [{file_count + 1:>3}/{total_files + 1}] tonecloud00_dur0ms_isi0ms.wav  (silence)")
    print(f"\nDone. Saved {file_count + 1} files to {out_dir}")


if __name__ == "__main__":
    main()
