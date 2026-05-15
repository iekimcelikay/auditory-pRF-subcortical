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
FREQ_RANGE       = (400, 1600, 3)    # (min_hz, max_hz, num_cfs)
SPECIES          = 'human'            # 'human' or 'cat'

# -- Stimulus parameters — sweep over all (duration, ISI) pairs ---------------
TONE_ON_MS       = (25, 50, 75, 150, 250, 350, 400, 500)  # ms
ISI_MS           = (100,) * len(TONE_ON_MS)                          # ms
TOTAL_DURATION   = 20      # s  — total sequence length
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

    cfs = calc_cfs(FREQ_RANGE, species=SPECIES)
    soundgen = SoundGen(SAMPLE_RATE, TAU_RAMP)

    total_files = len(TONE_ON_MS) * len(cfs)
    print(f"Saving {total_files} files to: {out_dir}")
    print(f"  CFs         : {cfs[0]:.1f} – {cfs[-1]:.1f} Hz  ({len(cfs)} Greenwood-spaced)")
    print(f"  Durations   : {TONE_ON_MS} ms")
    print(f"  ISIs        : {ISI_MS} ms")
    print()

    seq_num = START_SEQ_NUMBER
    file_count = 0

    for tone_ms, isi_ms in zip(TONE_ON_MS, ISI_MS):
        tone_s = tone_ms / 1000.0
        isi_s  = isi_ms  / 1000.0
        num_tones, _, _ = soundgen.calculate_num_tones(tone_s, isi_s, TOTAL_DURATION)

        for cf in cfs:
            sequence = soundgen.generate_sequence(
                freq=cf,
                num_harmonics=NUM_HARMONICS,
                tone_duration=tone_s,
                harmonic_factor=HARMONIC_FACTOR,
                dbspl=DBSPL,
                total_duration=TOTAL_DURATION,
                isi=isi_s,
                stereo=STEREO,
            )

            filename = (
                f"sequence{seq_num:02d}"
                f"_fc{cf:.0f}hz"
                f"_dur{tone_ms}ms"
                f"_isi{isi_ms}ms"
                f"_total{TOTAL_DURATION}sec"
                f"_numtones{num_tones}.wav"
            )
            save_sequence_as_wav(sequence, SAMPLE_RATE, str(out_dir / filename), subtype=WAV_SUBTYPE)

            file_count += 1
            print(f"  [{file_count:>3}/{total_files}] {filename}")
            seq_num += 1

    print(f"\nDone. Saved {file_count} files to {out_dir}")


if __name__ == "__main__":
    main()
