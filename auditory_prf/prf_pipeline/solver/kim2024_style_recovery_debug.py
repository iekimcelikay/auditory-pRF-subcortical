"""kim2024_style_recovery_debug.py
===================================
Minimal-grid trial run of kim2024_style_recovery.py's pipeline, for local
debugging only. Does NOT modify kim2024_style_recovery.py -- instead,
overrides its grid-size / sweep-size module globals in place before calling
its own (otherwise unmodified) main(), so the whole candidate-grid-build +
Monte Carlo recovery + accuracy pipeline runs in well under a minute on a
laptop.

IMPORTANT (see project discussion / conversation — a real bug hit here
once): ``kim2024_style_recovery.py`` imports ``ALPHA_GRID``/``W_GRID`` from
``grid_recovery.py`` via ``from grid_recovery import ALPHA_GRID, W_GRID`` —
that BINDS A COPY of those values in kim2024_style_recovery's own namespace
at import time, it is NOT a live reference back to grid_recovery's
attributes. Patching ``grid_recovery.ALPHA_GRID`` (as this module used to,
and as grid_recovery_debug.py correctly does for grid_recovery.py's OWN
functions) has NO EFFECT on kim2024_style_recovery.build_candidate_grid,
which reads ITS OWN module's ``ALPHA_GRID``/``W_GRID``/``TAU_GRID_MS``
globals. Every override below patches ``m.<name>`` (kim2024_style_recovery
itself), not ``gr.<name>``, for exactly this reason.

Every swept grid dimension is shrunk to 2-3 values (not 1), so indexing
across that axis still gets exercised rather than silently skipped. CF_index
is left untouched (it comes from the data, not a tunable grid, and is cheap
to loop over in full regardless). N_NOISE_REPEATS is also shrunk (Monte
Carlo repeats per voxel are back per project discussion) — 10,000 repeats
x 100 voxels would make even a "quick" debug run slow.

Outputs go to a separate kim2024_style_recovery_debug_results/ directory so
this never collides with a real run's results.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_debug

(``--no-capture-output`` matters — see kim2024_style_recovery.py's module
docstring: plain ``conda run`` withholds the child's stdout until it exits.)
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_recovery as m

m.TAU_GRID_MS  = [30, 50, 100]        # full grid: 9 values
m.ALPHA_GRID   = [1.6, 2.0, 3.2]       # full grid: 7 values
m.W_GRID       = [0.7, 0.8]            # full grid: 9 values

m.N_VOXELS        = 6                  # full run: 100 -- simplified: 6 independent voxels
m.N_NOISE_REPEATS = 100                # full run: 10,000 -- simplified: 100 MC repeats
m.DESIGN_SEEDS    = [1000, 1001]       # full run: 10 design seeds -- simplified: 2

# Separate base dir from a real run's "kim2024_style_recovery_results", but
# reuses m.RUN_TIMESTAMP (already stamped once at import time) rather than
# generating a second one, so there's a single run timestamp to reason about.
m.OUTPUT_DIR = Path("kim2024_style_recovery_debug_results")
m.RUN_DIR    = m.OUTPUT_DIR / m.RUN_TIMESTAMP


if __name__ == "__main__":
    m.main()
