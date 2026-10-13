"""grid_recovery_debug.py
==========================
Minimal-grid trial run of grid_recovery.py's pipeline, for local debugging
only. Does NOT modify grid_recovery.py — instead, overrides its grid-size
module globals in place before calling its own (otherwise unmodified)
main(), so the whole candidate-grid-build + Monte Carlo recovery + plotting
pipeline runs in well under a minute on a laptop.

Every swept dimension is shrunk to 2 values (not 1), so indexing across that
axis still gets exercised rather than silently skipped — see conversation
for why size-1 can hide bugs that size-2 won't. CF_index is left untouched
(it comes from the data, not a tunable grid, and is cheap to loop over in
full regardless).

Outputs go to a separate grid_recovery_debug_results/ directory so this
never collides with a real run's results.

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.grid_recovery_debug
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.grid_recovery as gr

gr.TAU_ON_GRID_MS  = [ 30, 50,  100] # full grid: 9 values
gr.TAU_OFF_GRID_MS = [ 30, 50, 100]        # full grid: 9 values
gr.ALPHA_GRID      = [1.6, 2.0, 3.2]       # full grid: 7 values
gr.W_GRID          = [0.7, 0.8]       # full grid: 9 values
gr.N_GT_COMBOS      = 5               # full run: 20 -- simplified: 5 ground truths
gr.N_NOISE_REPEATS  = 100             # full run: 10,000
gr.NOISE_VOXELS     = ["low"]         # full run: ["low", "mid", "high"] -- simplified: low only
gr.GT_SAMPLING_MODES = ["on_grid"]    # full run: ["on_grid", "off_grid"] -- simplified: on_grid only
gr.N_DESIGN_SEEDS   = 2               # full run: 10 -- simplified: 2 design seeds
gr.DESIGN_SEEDS     = [1000 + i for i in range(gr.N_DESIGN_SEEDS)]

# Separate base dir from a real run's "grid_recovery_results", but reuses
# gr.RUN_TIMESTAMP (already stamped once at import time) rather than
# generating a second one, so there's a single run timestamp to reason about.
# Per-(design-seed, mode, noise-level) paths, the candidate-grid path, and
# the comparison-plot path are all derived dynamically
# (gr.candidate_grid_path() / gr.recovery_output_paths() /
# gr.noise_comparison_plot_path()) from gr.RUN_DIR at call time, so
# overriding RUN_DIR alone is enough.
gr.OUTPUT_DIR          = Path("grid_recovery_debug_results")
gr.RUN_DIR             = gr.OUTPUT_DIR / gr.RUN_TIMESTAMP


if __name__ == "__main__":
    gr.main()
