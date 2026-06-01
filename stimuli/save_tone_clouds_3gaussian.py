"""
save_tone_clouds_3gaussian.py

Generates broadband noise-cloud sequences with 3 Gaussian frequency bands
whose centres are equally spaced on the Greenwood (cochlear-place) scale.

For each of the 3 band centres × each duration condition, one 20-second
sequence is saved in which every burst is an independent bandpass noise
realisation filtered to [centre / 2^sigma_oct, centre × 2^sigma_oct] Hz.
Each burst is unique (different random seed) so the sequence has no
repeating melodic structure.

Output filename convention:

    noisecloud{k:02d}_fc{center}hz_bw{sigma:.2f}oct_dur{D}ms_isi{I}ms_...

Usage
-----
    python stimuli/save_tone_clouds_3gaussian.py
"""

import sys
from pathlib import Path
from datetime import datetime

root = str(Path(__file__).resolve().parents[1])
stimuli_dir = str(Path(__file__).resolve().parent)
for p in (root, stimuli_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from auditory_prf.utils.stimulus_utils import calc_cfs
from soundgen import SoundGen
from save_sound import save_sequence_as_wav
from find_optimal_durations import find_closest_durations

# ==============================================================================
# CONFIG  — edit everything here
# ==============================================================================

# -- Frequency range: 3 Greenwood-equally-spaced band centres ------------------
FREQ_RANGE       = (450, 1600, 3)     # (min_hz, max_hz, num_bands)
SPECIES          = 'human'

# -- Gaussian bandwidth in octaves (±sigma_oct defines the bandpass edges) -----
SIGMA_OCT        = 0.5                # each band spans [CF/2^σ, CF×2^σ] Hz

# -- FIR filter length ---------------------------------------------------------
NUMTAPS          = 1001               # higher = sharper band edges, slower

# -- Stimulus timing -----------------------------------------------------------
TOTAL_DURATION   = 20                 # s
_TARGET_DURS_MS  = (35, 45, 60, 75, 100, 150, 250, 500)
ISI_MS_SINGLE    = 100
TONE_ON_MS       = find_closest_durations(
    _TARGET_DURS_MS,
    isi_ms=ISI_MS_SINGLE,
    seq_dur_s=TOTAL_DURATION,
    dur_max_ms=max(_TARGET_DURS_MS),
)
ISI_MS           = (ISI_MS_SINGLE,) * len(TONE_ON_MS)

# -- Acoustic parameters -------------------------------------------------------
DBSPL            = 65
TAU_RAMP         = 0.005              # s — onset/offset ramp
SAMPLE_RATE      = 100_000            # Hz
STEREO           = True

# -- Reproducibility -----------------------------------------------------------
BASE_SEED        = 42

# -- Output --------------------------------------------------------------------
BASE_OUT_DIR     = Path(__file__).parent / "produced"
RUN_PREFIX       = "noiseclouds3g"
WAV_SUBTYPE      = "FLOAT"

# ==============================================================================
# HELPERS
# ==============================================================================


def _band_edges_hz(center_hz: float, sigma_oct: float) -> tuple[float, float]:
    """Return (lowcut_hz, highcut_hz) for ±sigma_oct around center_hz."""
    return center_hz / (2 ** sigma_oct), center_hz * (2 ** sigma_oct)


def _cond_id(
    band_idx: int,
    center_hz: float,
    sigma_oct: float,
    dur_ms: float,
    isi_ms: float,
) -> str:
    return (
        f"noisecloud{band_idx:02d}"
        f"_fc{int(round(center_hz))}hz"
        f"_bw{sigma_oct:.2f}oct"
        f"_dur{int(round(dur_ms))}ms"
        f"_isi{int(round(isi_ms))}ms"
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = BASE_OUT_DIR / f"{RUN_PREFIX}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    centers_hz = calc_cfs(FREQ_RANGE, species=SPECIES)
    soundgen = SoundGen(SAMPLE_RATE, TAU_RAMP)
    n_durs = len(TONE_ON_MS)
    total_files = n_durs * len(centers_hz)

    print(f"Saving {total_files} noise-cloud files to: {out_dir}")
    print(f"  Band centres : {[f'{c:.1f}' for c in centers_hz]} Hz")
    print(f"  Bandwidth    : ±{SIGMA_OCT} oct  "
          f"(bands: {[f'{_band_edges_hz(c, SIGMA_OCT)[0]:.0f}–{_band_edges_hz(c, SIGMA_OCT)[1]:.0f}' for c in centers_hz]} Hz)")
    print(f"  Durations    : {TONE_ON_MS} ms")
    print()

    file_count = 0
    for dur_idx, (tone_ms, isi_ms) in enumerate(zip(TONE_ON_MS, ISI_MS)):
        tone_s = tone_ms / 1000.0
        isi_s  = isi_ms  / 1000.0
        num_tones, _, _ = soundgen.calculate_num_tones(tone_s, isi_s, TOTAL_DURATION)

        for band_idx, center_hz in enumerate(centers_hz):
            lowcut_hz, highcut_hz = _band_edges_hz(center_hz, SIGMA_OCT)
            seed = BASE_SEED + band_idx * n_durs + dur_idx

            sequence = soundgen.generate_bandpass_noise_sequence(
                lowcut_hz=lowcut_hz,
                highcut_hz=highcut_hz,
                tone_duration=tone_s,
                dbspl=DBSPL,
                total_duration=TOTAL_DURATION,
                isi=isi_s,
                numtaps=NUMTAPS,
                base_seed=seed,
                stereo=STEREO,
            )

            cond_id = _cond_id(band_idx + 1, center_hz, SIGMA_OCT, tone_ms, isi_ms)
            filename = f"{cond_id}_total{TOTAL_DURATION}sec_numtones{num_tones}.wav"
            save_sequence_as_wav(
                sequence, SAMPLE_RATE, str(out_dir / filename), subtype=WAV_SUBTYPE
            )

            file_count += 1
            print(f"  [{file_count:>3}/{total_files}] {filename}")

    # Silence WAV — null trial
    n_silence = int(SAMPLE_RATE * TOTAL_DURATION)
    silence = np.zeros((n_silence, 2) if STEREO else (n_silence,), dtype=np.float32)
    save_sequence_as_wav(
        silence, SAMPLE_RATE, str(out_dir / "noisecloud00_dur0ms_isi0ms.wav"),
        subtype=WAV_SUBTYPE,
    )
    file_count += 1
    print(f"  [{file_count:>3}/{total_files + 1}] noisecloud00_dur0ms_isi0ms.wav  (silence)")

    print(f"\nDone. Saved {file_count} files to {out_dir}")


if __name__ == "__main__":
    main()
