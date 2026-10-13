"""kim2024_style_recovery_debug.py
===================================
Minimal-grid trial run of kim2024_style_recovery.py's pipeline, for local
debugging only. Does NOT modify kim2024_style_recovery.py or grid_recovery.py
-- instead, overrides their grid-size / sweep-size module globals in place
before calling kim2024_style_recovery's own (otherwise unmodified) main(), so
the whole candidate-grid-build + independent-voxel recovery + accuracy
pipeline runs in well under a minute on a laptop. Same pattern as
grid_recovery_debug.py / duration_recovery_debug.py.

Every swept grid dimension is shrunk to 2-3 values (not 1), so indexing
across that axis still gets exercised rather than silently skipped. CF_index
is left untouched (it comes from the data, not a tunable grid, and is cheap
to loop over in full regardless).

Outputs go to a separate kim2024_style_recovery_debug_results/ directory so
this never collides with a real run's results.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_debug

(``--no-capture-output`` matters — see kim2024_style_recovery.py's module
docstring: plain ``conda run`` withholds the child's stdout until it exits.)
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.grid_recovery as gr
import auditory_prf.prf_pipeline.solver.kim2024_style_recovery as m

gr.TAU_ON_GRID_MS  = [30, 50, 100]    # full grid: 9 values
gr.TAU_OFF_GRID_MS = [30, 50, 100]    # full grid: 9 values
gr.ALPHA_GRID      = [1.6, 2.0, 3.2]  # full grid: 7 values
gr.W_GRID          = [0.7, 0.8]       # full grid: 9 values

m.N_VOXELS     = 6                    # full run: 100 -- simplified: 6 independent voxels
m.DESIGN_SEEDS = [1000, 1001]         # full run: 10 design seeds -- simplified: 2

# Separate base dir from a real run's "kim2024_style_recovery_results", but
# reuses m.RUN_TIMESTAMP (already stamped once at import time) rather than
# generating a second one, so there's a single run timestamp to reason about.
m.OUTPUT_DIR = Path("kim2024_style_recovery_debug_results")
m.RUN_DIR    = m.OUTPUT_DIR / m.RUN_TIMESTAMP


if __name__ == "__main__":
    m.main()
