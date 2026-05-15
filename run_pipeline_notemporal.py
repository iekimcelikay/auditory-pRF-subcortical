"""
Run full_pipeline_notemporal with 1000 run designs.

Each run: sample 8 temporal conditions from TONE_ON_MS × 3 frequencies
= 24 unique conditions + null trials.

Local (all CFs sequentially):
    python run_pipeline_notemporal.py

Single CF (for SLURM array):
    python run_pipeline_notemporal.py --cf 7
"""

import argparse
import numpy as np
from pathlib import Path

from auditory_prf.utils.stimulus_utils import calc_cfs
from auditory_prf.stimuli.soundgen import SoundGen
from auditory_prf.prf_pipeline.run_assembly import make_seq_id_fn, generate_run_design
from auditory_prf.prf_pipeline.full_pipeline_notemporal import run_pipeline

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("./models_output/dipc_test_280514_01")
OUTPUT_DIR  = Path("./output/prf_notemporal_280514")

# ── cochlear CFs (indices into the 30-CF cochlear model) ──────────────────────
CF_INDICES  = list(range(30))

# ── model params ──────────────────────────────────────────────────────────────
ALPHA = 8.0

# ── stimulus params (must match WAV files in RESULTS_DIR) ────────────────────
ALL_DURATIONS    = (25, 75, 100, 150, 250, 300, 350, 400, 450, 500)  # ms
ISI_MS           = 100                  # ms — fixed for all conditions
FREQ_RANGE       = (400, 1600, 3)       # matches fc400hz / fc830hz / fc1600hz
TRIAL_DURATION_S = 20.0
OPENING_BLANK_S  = 10.0
ITI_RANGE_S      = (0, 0)
NULL_FRACTION    = 0.25

# ── run design params ─────────────────────────────────────────────────────────
N_DESIGNS            = 1000
N_CONDITIONS_PER_RUN = 8   # temporal conditions sampled per run (out of 10)
BASE_SEED            = 42

# ── HRF / BOLD ────────────────────────────────────────────────────────────────
TOTAL_RUN_DUR_S = 720.0
TR_S            = 1.6


def build_run_designs() -> list:
    """Generate all 1000 run designs (shared across CFs)."""
    desired_freqs = calc_cfs(FREQ_RANGE, species='human')
    sound_gen     = SoundGen(48000, tau=0.005)
    seq_id_fn     = make_seq_id_fn(FREQ_RANGE, TRIAL_DURATION_S, sound_gen)

    print(f"Building {N_DESIGNS} run designs "
          f"({N_CONDITIONS_PER_RUN} durations × {len(desired_freqs)} freqs "
          f"= {N_CONDITIONS_PER_RUN * len(desired_freqs)} active trials/run)...")

    run_designs = []
    for i in range(N_DESIGNS):
        rng = np.random.default_rng(BASE_SEED + i)

        sampled_durations = rng.choice(ALL_DURATIONS, size=N_CONDITIONS_PER_RUN, replace=False)

        stimuli = [(int(dur), ISI_MS, freq)
                   for dur in sampled_durations
                   for freq in desired_freqs]
        n_null      = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
        base_trials = stimuli + [(0, 0, None)] * n_null

        run_designs.append(generate_run_design(
            base_trials, seq_id_fn,
            trial_duration_s = TRIAL_DURATION_S,
            opening_blank_s  = OPENING_BLANK_S,
            iti_range_s      = ITI_RANGE_S,
            seed             = int(rng.integers(0, 2**32)),
        ))

    print(f"Done — {len(run_designs)} designs, "
          f"{len(run_designs[0])} trials each (including nulls).")
    return run_designs


def process_cf(cf: int, run_designs: list, n_workers: int = 1) -> None:
    print(f"\n{'='*50}  CF {cf}/{max(CF_INDICES)}  {'='*50}")
    run_pipeline(
        cf              = cf,
        results_dir     = RESULTS_DIR,
        output_dir      = OUTPUT_DIR,
        alpha           = ALPHA,
        freq_range      = FREQ_RANGE,
        trial_duration_s= TRIAL_DURATION_S,
        opening_blank_s = OPENING_BLANK_S,
        total_run_dur_s = TOTAL_RUN_DUR_S,
        tr_s            = TR_S,
        run_designs     = run_designs,
        n_workers       = n_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run notemporal pRF pipeline. "
                    "Omit --cf to process all CFs sequentially (local use)."
    )
    parser.add_argument("--cf", type=int, default=None,
                        help="Single CF index to process (for SLURM array). "
                             "Omit to run all CFs sequentially.")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default=None)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Worker processes for Phase 2 assembly. "
                             "Set to $SLURM_CPUS_PER_TASK on cluster.")
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    run_designs = build_run_designs()

    if args.cf is not None:
        process_cf(args.cf, run_designs, n_workers=args.n_workers)
    else:
        for cf in CF_INDICES:
            process_cf(cf, run_designs, n_workers=args.n_workers)
