# Parameter Recovery Test — Subcortical AdapTrans+BOLD Pipeline

**Run:** `20261007_130502`
**Pipeline:** `auditory_prf.prf_pipeline.solver.grid_recovery_parallel_per_alpha`

## Objective

Validate whether the grid-search / template-matching procedure used to fit the subcortical pRF model (CF tuning + power-law sharpening + AdapTrans onset/offset filtering) can recover known ground-truth parameters from simulated BOLD data, before applying it to real scans.

## Method

For each of a set of known "ground truth" parameter combinations — `CF_index`, `tau_on_ms` and `tau_off_ms` (AdapTrans onset/offset time constants), `alpha` (power-law sharpening exponent), and `w` (adaptation weight) — a noiseless BOLD signal was simulated. Independent `PmNoise`-based noisy realizations of that signal were then correlated (Pearson *r*) against a joint candidate grid of 153,090 noiseless model predictions (30 CF × 9 `tau_on` × 9 `tau_off` × 7 `alpha` × 9 `w`); the best-correlating candidate was taken as the recovered parameter estimate.

Note: `tau_on`/`tau_off` are treated as **free parameters, independent of CF** — the forward model does not tie them to characteristic frequency via any formula (e.g. the Willmore et al. 2016 relationship, τ(f) = 500 − 105·log₁₀(f_Hz), is not used here; see `run_assembly.py`'s `assemble_run_bold` docstring: "Free parameter — no CF-tau relationship assumed"). Ground-truth `cf_index`, `tau_on_ms`, `tau_off_ms`, `alpha`, and `w` were therefore sampled independently of each other for every combination.

This was repeated for:
- **20 ground-truth combinations**, drawn jointly across all 5 parameters (not one-at-a-time), under two sampling modes: **on-grid** (true values coincide exactly with a candidate grid point) and **off-grid** (true `tau_on`/`tau_off`/`alpha`/`w` drawn continuously, testing grid-resolution/quantization error).
- **10,000 independent noise draws per ground truth**, at three BOLD noise levels (**low / mid / high**, `PmNoise` presets), using the same noise draws across levels so that differences are attributable to noise amplitude alone, not to which random draws happened to occur.

## Results

Mean absolute recovery bias per parameter, as a percentage of that parameter's candidate-grid range:

| Condition | `cf_index` | `tau_on_ms` | `tau_off_ms` | `alpha` | `w` |
|---|---|---|---|---|---|
| on-grid, low noise | 0% | **23%** | **17%** | 1% | 6% |
| on-grid, mid noise | 2% | **32%** | **21%** | 5% | 13% |
| on-grid, high noise | 3% | **34%** | 19% | 9% | 20% |
| off-grid, low noise | 1% | **15%** | **21%** | 4% | 8% |
| off-grid, mid noise | 1% | **19%** | **29%** | 10% | 15% |
| off-grid, high noise | 3% | **23%** | **29%** | 16% | 21% |

**`cf_index`, `alpha`, and `w` recover well** (bias stays under ~20% of range even at the highest noise level tested). **`tau_on_ms` and `tau_off_ms` do not** — bias is already 15–23% of their range at the *lowest* noise level, meaning the problem is not primarily a noise-robustness issue; it persists even under near-ideal conditions.

## Why: a mechanistic check, not just a statistical one

Noiseless candidate templates differing only in `tau_off` (30 ms vs. 50 ms vs. 100 ms — more than a 3× range) correlate with each other at **r = 0.996–0.999**, before any noise is added. Decomposing the signal confirms why:
- The AdapTrans **ON** response component is completely unaffected by `tau_off` and contributes roughly half the total signal variance, diluting whatever distinction the OFF component carries.
- Even isolating the **OFF** component alone, `tau_off = 30 ms` vs. `100 ms` still correlates at **r = 0.978** — because the HRF convolution (~32 s kernel width) smooths out timing differences on the order of tens of milliseconds almost completely before the signal reaches the sampled BOLD timecourse (`TR = 1.6 s`).

This is a property of the forward model and acquisition parameters (TR, HRF), not of the grid-search procedure, the noise level, or the candidate grid's resolution — a finer `tau_on`/`tau_off` grid would not resolve it, and could make it marginally worse (more near-identical neighboring candidates for noise to confuse).

## Figures

**Recovery across noise levels, on-grid ground truths** (one point per noisy repeat, pooled across 20 ground truths; color = noise level):

![on-grid noise comparison](grid_recovery_noise_comparison_on_grid_20261007_130502.png)

**Recovery across noise levels, off-grid ground truths:**

![off-grid noise comparison](grid_recovery_noise_comparison_off_grid_20261007_130502.png)

In both figures, `cf_index`/`alpha`/`w` show tight clouds hugging the diagonal (true = recovered); `tau_on_ms`/`tau_off_ms` show wide, poorly-structured scatter even in the green ("low noise") points.

## Conclusion

Under the current acquisition parameters (`TR = 1.6 s`, standard subcortical HRF), this template-matching procedure reliably recovers `CF_index`, `alpha`, and `w`, but **cannot reliably recover `tau_on`/`tau_off`** — independent of noise level or candidate-grid resolution. This should be treated as a finding about the model/acquisition's identifiability limits, not a bug in the fitting code (correctness of the implementation was separately verified against a reference serial implementation, bit-for-bit).

## Full per-condition data

Per-ground-truth bias tables and individual scatter plots for all 6 (sampling-mode × noise-level) conditions are included alongside this file in the same run directory (`grid_recovery_table_*.csv`, `grid_recovery_scatter_*.png`).
