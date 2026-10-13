"""kim2024_style_duration_recovery_debug.py
=============================================
Minimal-grid trial run of kim2024_style_duration_recovery.py's pipeline, for
local debugging only. Does NOT modify kim2024_style_duration_recovery.py or
duration_recovery.py -- overrides their grid-size / sweep-size module
globals in place before calling kim2024_style_duration_recovery's own
(otherwise unmodified) main().

IMPORTANT (see kim2024_style_recovery_debug.py's module docstring — the
same lesson applies here): ``kim2024_style_duration_recovery.py`` REUSES
``duration_recovery.py``'s ``build_candidate_grid``/``sample_ground_truths_off_grid``/
``run_recovery`` function objects UNCHANGED (not reimplemented) — those
functions read ``PREF_DUR_GRID_MS``/``SIGMA_DUR_GRID_MS``/``ALPHA_GRID``/
``N_NOISE_REPEATS`` from ``duration_recovery.py``'s OWN module globals at
call time, so overrides below patch ``dr.<name>`` (duration_recovery) for
those. ``N_VOXELS``/``DESIGN_SEEDS`` are this script's OWN names (not
reused from elsewhere), so those patch ``m.<name>`` (kim2024_style_duration_recovery).

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_debug

(``--no-capture-output`` matters — see kim2024_style_recovery.py's module
docstring: plain ``conda run`` withholds the child's stdout until it exits.)
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.duration_recovery as dr
import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery as m

dr.PREF_DUR_GRID_MS  = [34.89, 496.43]   # full grid: 8 values (all of TONE_ON_MS)
dr.SIGMA_DUR_GRID_MS = [20.0, 100.0]     # full grid: 44 values
dr.ALPHA_GRID        = [1.0, 8.0]        # full grid: 7 values
dr.N_NOISE_REPEATS   = 100               # full run: 10,000 -- simplified: 100 MC repeats

m.N_VOXELS     = 6                       # full run: 100 -- simplified: 6 independent voxels
m.DESIGN_SEEDS = [1000, 1001]            # full run: 10 design seeds -- simplified: 2

# Separate base dir from a real run's "kim2024_style_duration_recovery_results",
# but reuses m.RUN_TIMESTAMP (already stamped once at import time).
m.OUTPUT_DIR = Path("kim2024_style_duration_recovery_debug_results")
m.RUN_DIR    = m.OUTPUT_DIR / m.RUN_TIMESTAMP


if __name__ == "__main__":
    m.main()
