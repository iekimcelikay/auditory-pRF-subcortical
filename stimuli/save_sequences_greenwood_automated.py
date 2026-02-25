"""
save_sequences_greenwood_automated.py

Generates and saves a tone sequence .wav file for each characteristic frequency
(CF) in a Greenwood-spaced CF array that matches the model CFs used in
CochleaConfig / ToneConfig.

Each run creates a new timestamped subfolder under BASE_OUT_DIR so that
consecutive runs never overwrite previous outputs.

Usage
-----
    python stimuli/save_sequences_greenwood_automated.py

All user-adjustable parameters are in the CONFIG block below.
"""

import sys
from pathlib import Path
from datetime import datetime

# --- ensure project root and stimuli/ are importable --------------------------
root = str(Path(__file__).resolve().parents[1])
stimuli_dir = str(Path(__file__).resolve().parent)
for p in (root, stimuli_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from auditory_prf.utils.stimulus_utils import calc_cfs
from soundgen import SoundGen
from save_sound import save_sequence_as_wav

# ==============================================================================
# CONFIG  — edit everything here
# ==============================================================================

# -- Frequency / CF (mirrors ToneConfig.freq_range / CochleaConfig) -----------
FREQ_RANGE       = (125.0, 2500.0, 40)   # (min_hz, max_hz, num_cfs)
SPECIES          = 'human'               # 'human' or 'cat'

# -- Stimulus parameters -------------------------------------------------------
TONE_DURATION    = 0.267    # s  — duration of a single tone
ISI              = 0.133    # s  — inter-stimulus interval
TOTAL_DURATION   = 1.5      # s  — total sequence length
DBSPL            = 60       # dB SPL
NUM_HARMONICS    = 1
HARMONIC_FACTOR  = 1
TAU_RAMP         = 0.005    # s  — onset/offset ramp
SAMPLE_RATE      = 100000   # Hz
STEREO           = True     # True → (N, 2) stereo wav; False → (N,) mono

# -- Output --------------------------------------------------------------------
BASE_OUT_DIR     = Path(__file__).parent / "produced"
RUN_PREFIX       = ""    # subfolder: produced/{RUN_PREFIX}_{YYYYMMDD_HHMM}/
START_SEQ_NUMBER = 1            # sequence numbers count up from this value
WAV_SUBTYPE      = "FLOAT"       # soundfile subtype; "PCM_16" for integer 16-bit

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # Create a new timestamped run folder on every execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = BASE_OUT_DIR / f"{RUN_PREFIX}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate Greenwood-spaced CF array (identical to CochleaConfig / ToneConfig)
    cfs = calc_cfs(FREQ_RANGE, species=SPECIES)

    soundgen = SoundGen(SAMPLE_RATE, TAU_RAMP)
    num_tones, _, _ = soundgen.calculate_num_tones(TONE_DURATION, ISI, TOTAL_DURATION)

    print(f"Saving {len(cfs)} files to: {out_dir}")
    print(f"  CFs : {cfs[0]:.1f} – {cfs[-1]:.1f} Hz  ({len(cfs)} Greenwood-spaced)")
    print(f"  Sequence numbers: {START_SEQ_NUMBER} – {START_SEQ_NUMBER + len(cfs) - 1}")
    print()

    for i, cf in enumerate(cfs):
        seq_num = START_SEQ_NUMBER + i

        sequence = soundgen.generate_sequence(
            freq=cf,
            num_harmonics=NUM_HARMONICS,
            tone_duration=TONE_DURATION,
            harmonic_factor=HARMONIC_FACTOR,
            dbspl=DBSPL,
            total_duration=TOTAL_DURATION,
            isi=ISI,
            stereo=STEREO,
        )

        filename = (
            f"sequence{seq_num:02d}"
            f"_fc{cf:.0f}hz"
            f"_dur{TONE_DURATION * 1000:.0f}ms"
            f"_isi{ISI * 1000:.0f}ms"
            f"_total{TOTAL_DURATION}sec"
            f"_numtones{num_tones}.wav"
        )
        out_file = out_dir / filename

        save_sequence_as_wav(sequence, SAMPLE_RATE, str(out_file), subtype=WAV_SUBTYPE)
        print(f"  [{i + 1:>3}/{len(cfs)}] {filename}")

    print(f"\nDone. Saved {len(cfs)} files to {out_dir}")


if __name__ == "__main__":
    main()
