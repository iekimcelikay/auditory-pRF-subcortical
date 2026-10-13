"""duration_recovery_debug.py
==============================
Minimal-grid trial run of duration_recovery.py's pipeline, for local
debugging only. Does NOT modify duration_recovery.py -- overrides its
grid-size module globals in place before calling its own main(), same
pattern as grid_recovery_debug.py.

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.duration_recovery_debug
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.duration_recovery as dr

dr.PREF_DUR_GRID_MS  = [34.89, 496.43]     # full grid: 8 values (all of TONE_ON_MS)
dr.SIGMA_DUR_GRID_MS = [20.0, 100.0]       # full grid: 44 values
dr.ALPHA_GRID        = [1.0, 8.0]          # full grid: 7 values
dr.N_GT_COMBOS       = 5                  # full run: 20
dr.N_NOISE_REPEATS   = 100                # full run: 10,000
dr.NOISE_VOXELS      = ["low"]            # full run: ["low", "mid", "high"]
dr.GT_SAMPLING_MODES = ["on_grid"]        # full run: ["on_grid", "off_grid"]

dr.OUTPUT_DIR          = Path("duration_recovery_debug_results")
dr.RUN_DIR             = dr.OUTPUT_DIR / dr.RUN_TIMESTAMP
dr.CANDIDATE_GRID_PATH = dr.RUN_DIR / f"candidate_grid_{dr.RUN_TIMESTAMP}.npz"


if __name__ == "__main__":
    dr.main()
