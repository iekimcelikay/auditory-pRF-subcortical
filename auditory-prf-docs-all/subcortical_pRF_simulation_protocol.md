# Subcortical Temporal pRF Modelling — Simulation Protocol

*Version 1.0 — April 2026*

---

## Table of Contents

1. [Glossary](#1-glossary)
2. [Forward Model Pipeline](#2-forward-model-pipeline)
3. [The Four-Model Hierarchy](#3-the-four-model-hierarchy)
4. [Simulation Protocol](#4-simulation-protocol)
   - 4.1 [Overview and Purpose](#41-overview-and-purpose)
   - 4.2 [General Procedure](#42-general-procedure)
   - 4.3 [Simulation 1 — Model 1: Spectral Only](#43-simulation-1--model-1-spectral-only)
   - 4.4 [Simulation 2 — Model 2: Spectral + Duration](#44-simulation-2--model-2-spectral--duration)
   - 4.5 [Simulation 3 — Model 4: Spectral + AdapTrans](#45-simulation-3--model-4-spectral--adaptrans)
   - 4.6 [Simulation 4 — Model 3: Spectral + Duration + AdapTrans](#46-simulation-4--model-3-spectral--duration--adaptrans)
   - 4.7 [The Coupling Diagnostic (Model 3 only)](#47-the-coupling-diagnostic-model-3-only)
   - 4.8 [Model Misspecification Tests](#48-model-misspecification-tests)
5. [Severity Thresholds and Remedies](#5-severity-thresholds-and-remedies)
6. [Recommended Execution Order](#6-recommended-execution-order)

---

## 1. Glossary

**Free parameter**
A quantity the optimizer is allowed to adjust during model fitting. Each voxel has its own set of free parameter values — they are not shared across voxels. The optimizer searches for the values that make the model's predicted BOLD timecourse as close as possible to the measured one.

**Fixed parameter / Fixed input**
A quantity that is computed directly from the stimulus or set from prior literature, and is not adjusted during fitting. In this pipeline, auditory nerve (AN) firing rates and boxcar timing are fixed inputs derived from the stimulus design.

**Forward model**
The full computational pipeline that takes a set of parameter values and produces a predicted BOLD timecourse. Running the forward model once generates one prediction. The optimizer runs it many times, adjusting parameters each time to reduce the prediction error.

**Model fitting / Solving**
The process of finding the free parameter values that minimise the difference between the forward model's predicted BOLD and the actually measured BOLD. In this project, fitting is done by gradient descent in PyTorch, using the loss function described in Section 2.

**Solver**
The component that performs model fitting. Given a BOLD timecourse and a stimulus sequence, the solver runs the optimizer and returns the best-fitting parameter values. Analogous to the "Solver" in Kim et al. (2024), though implemented here with gradient descent rather than BADS.

**Synthesizer**
The component that generates synthetic (simulated) BOLD timecourses from known ground-truth parameters. Used during the simulation phase to test whether the solver can recover parameters it was not shown. Analogous to the "Synthesizer" in Kim et al. (2024).

**Ground-truth parameters**
The known parameter values used by the synthesizer to generate a synthetic BOLD timecourse. When the solver is run on that timecourse, recovered parameters should ideally match the ground-truth ones.

**Parameter recovery**
The process of checking how well the solver's estimated parameters match the ground-truth parameters used to generate the synthetic data. Recovery is quantified by MAPE (median absolute percentage error) and Pearson r.

**MAPE (Median Absolute Percentage Error)**
A recovery metric. For each simulated voxel, the absolute percentage difference between estimated and ground-truth parameter value is computed. The median of these across all voxels is the MAPE. Smaller is better. Kim et al. reported MAPEs of ~5–13% for well-recovered spatial parameters.

**Parameter coupling**
A situation where the recovery error in one parameter is systematically related to the true value of a different parameter. Coupling means parameters are not independently identifiable — the optimizer trades one off against the other. This is the key concern for Model 3, where `pref_dur` and `τ_ON/τ_OFF` may couple.

**Identifiability**
A model is identifiable if, given enough data, there is only one set of parameter values that produces any given BOLD timecourse. If two different parameter combinations produce identical predictions, the model is not identifiable and the solver cannot recover parameters reliably.

**Loss function**
The mathematical quantity the optimizer minimises. In this project: Gaussian negative log-likelihood (NLL) plus MAP prior terms on `τ` and `w`. Under Gaussian noise assumptions, NLL is equivalent to residual sum of squares (RSS). Adding MAP priors is equivalent to L2 regularisation centred on biologically motivated values.

**MAP prior**
Maximum A Posteriori prior. A soft constraint added to the loss function that penalises parameter values far from a biologically motivated initialisation. For `τ`, the prior is centred on the Rançon et al. Eq. 5 values (guinea pig IC). For `w`, the prior is centred on 0.75. The width (σ) of the prior determines how tightly parameters are constrained. Narrower σ → stronger constraint → less risk of coupling, but potentially biased estimates if the prior is wrong.

**AdapTrans**
The adapting-transient filter from Rançon et al. (2024). A pair of linear filters (ON and OFF channels) that respond to changes in their input — onsets and offsets — rather than sustained levels. Controlled by time constant `τ` (how quickly past input is forgotten) and `w` (balance between sustained and transient response). Applied per voxel after the boxcar reconstruction.

**HRF (Haemodynamic Response Function)**
The stereotyped BOLD signal produced by a brief neural event. Because the BOLD signal is sluggish (~12 s duration), predicted neural activity must be convolved with the HRF to generate a predicted BOLD timecourse. In real data, the HRF is fitted per voxel. In simulations, a canonical subcortical HRF is used.

**CF (Characteristic Frequency)**
The frequency to which an auditory neuron (or voxel) responds most strongly. Each voxel has one CF, recovered either from a separate tonotopy localiser or as a free parameter.

---

## 2. Forward Model Pipeline

The pipeline maps stimulus + parameters → predicted BOLD. Every stage must be implemented in PyTorch to preserve the computation graph for gradient-based fitting.

```
Stimulus (tone sequence)
        │
        ▼
Stage 1: Auditory Nerve (AN) Model
  → Pre-computed firing rates per CF × time
  → Stored as .npz; fixed input (no gradient)
        │
        ▼
Stage 2: Spectral Sharpening
  → r_sharp = r_AN ^ α
  → Free parameter: α
  → Mimics lateral inhibition (CN → IC)
        │
        ▼
Stages 3 & 5: Chunking & Boxcar Reconstruction
  → For each tone presentation: compute mean firing rate over tone-ON window
  → Reconstruct a continuous boxcar train:
       each boxcar has width = tone duration, amplitude = mean rate
  → Fixed computation; no free parameters
        │
        ▼
Stage 4: Duration Gaussian  [Models 2 and 3 only]
  → Scale each tone's boxcar amplitude by:
       G(d) = exp(-(d - pref_dur)² / (2 × σ_dur²))
  → Free parameters: pref_dur, σ_dur
        │
        ▼
Stage 6: AdapTrans ON/OFF Filters  [Models 3 and 4 only]
  → Apply ON and OFF adapting-transient filters to the boxcar train
  → ON channel: responds positively to onsets, negatively to offsets
  → OFF channel: responds positively to offsets, negatively to onsets
  → Sustained component (1 - w) passes through unchanged
  → Free parameters: w, ON weight, OFF weight, τ_ON, τ_OFF
        │
        ▼
Stage 7: HRF Convolution
  → Convolve neural timecourse with HRF → predicted BOLD
  → HRF fixed in simulation; fitted per voxel in real data
        │
        ▼
Predicted BOLD timecourse  →  Compare to measured BOLD  →  Loss
```

### Loss Function

```
L(θ) = (1/2σ²) Σ_t (y_t - ŷ_t)²
      + λ_τ  × Σ_CF [(τ_ON - τ_init(CF))² / (2σ_τ²) + (τ_OFF - τ_init(CF)×1.5)² / (2σ_τ²)]
      + λ_w  × (w - 0.75)² / (2σ_w²)
```

Where `τ_init(CF) = 500 - 105 × log(CF)` (Rançon et al. Eq. 5, guinea pig IC initialisation).

---

## 3. The Four-Model Hierarchy

Models are tested from simplest to most complex. Each model is a strict extension of the previous one. Parameters are **per voxel** — each voxel is characterised by its own set of values, analogous to how each voxel in visual pRF mapping has its own x, y, and σ.

---

### Model 1 — Spectral Only

> *Does the voxel have a preferred frequency? Can the AN model weighted by CF explain BOLD?*

**Stages active:** 1, 2, 3, 5, 7

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–4000 Hz |
| α | Spectral sharpening exponent | TBD |

**Total free parameters per voxel: 2**

No explicit duration tuning; no onset/offset responses. The BOLD response to each frequency step is determined solely by how well the stimulus frequency matches the voxel's CF.

---

### Model 2 — Spectral + Duration

> *Does the voxel additionally prefer tones of a specific duration?*

**Stages active:** 1, 2, 3, 4, 5, 7

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–4000 Hz |
| α | Spectral sharpening exponent | TBD |
| pref_dur | Voxel's preferred tone duration | 54–5000 ms |
| σ_dur | Width of duration tuning | 50–2000 ms |

**Total free parameters per voxel: 4**

Duration tuning is explicit and parametric via a Gaussian. There are no onset/offset transient responses — the neural drive is purely sustained (boxcar shaped). This is the cleanest test of whether duration selectivity exists at all in the data, without any confound from adaptation dynamics.

---

### Model 3 — Spectral + Duration + AdapTrans

> *Can onset/offset dynamics and explicit duration tuning together better explain the data?*

**Stages active:** 1, 2, 3, 4, 5, 6, 7

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–4000 Hz |
| α | Spectral sharpening exponent | TBD |
| pref_dur | Voxel's preferred tone duration | 54–5000 ms |
| σ_dur | Width of duration tuning | 50–2000 ms |
| w | Sustained vs. transient balance (0=sustained, 1=transient) | 0–1 |
| ON weight | Scaling of onset response | TBD |
| OFF weight | Scaling of offset response | TBD |
| τ_ON | Time constant of onset filter | 10–500 ms |
| τ_OFF | Time constant of offset filter | 10–500 ms |

**Total free parameters per voxel: 9**

This is the most complex model and the one requiring the most careful simulation validation. The key concern is **parameter coupling** between `pref_dur` and `τ_ON/τ_OFF`, because both mechanisms produce duration-dependent response profiles (see Section 4.7).

---

### Model 4 — Spectral + AdapTrans (no Duration Gaussian)

> *Can onset/offset dynamics alone — without an explicit duration Gaussian — account for duration selectivity?*

**Stages active:** 1, 2, 3, 5, 6, 7

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–4000 Hz |
| α | Spectral sharpening exponent | TBD |
| w | Sustained vs. transient balance | 0–1 |
| ON weight | Scaling of onset response | TBD |
| OFF weight | Scaling of offset response | TBD |
| τ_ON | Time constant of onset filter | 10–500 ms |
| τ_OFF | Time constant of offset filter | 10–500 ms |

**Total free parameters per voxel: 7**

This model tests the hypothesis that duration tuning in IC emerges implicitly from adaptation dynamics — specifically, from the overlap of ON and OFF AdapTrans responses as a function of tone duration — rather than from a separate explicit duration-tuning mechanism. It is the foil to Model 2: comparing their fits tells you which mechanism better explains the data.

---

### Model Comparison Summary

| Model | Stages | Free params | Key question |
|---|---|---|---|
| 1: Spectral only | 1,2,3,5,7 | 2 | Is there tonotopic organisation? |
| 2: + Duration | 1,2,3,4,5,7 | 4 | Is there explicit duration tuning? |
| 3: + Duration + AdapTrans | 1,2,3,4,5,6,7 | 9 | Do both mechanisms jointly improve fit? |
| 4: + AdapTrans only | 1,2,3,5,6,7 | 7 | Does adaptation alone explain duration selectivity? |

Comparing Model 2 vs Model 4 is the key scientific contrast: explicit duration tuning (Gaussian) vs. implicit duration selectivity (AdapTrans overlap dynamics). Model 3 asks whether combining them adds further explanatory power. Model 1 is the baseline.

---

## 4. Simulation Protocol

### 4.1 Overview and Purpose

The simulation phase is completed before any real data are collected. Its purpose is to verify that the solver can reliably recover ground-truth parameter values from synthetic BOLD timecourses. This follows Kim et al. (2024) Figure 4, which showed that parameter recovery accuracy must be established in simulation before results from real data can be interpreted.

A separate simulation is run for each model. Each simulation consists of two components:

- **Synthesizer:** generates synthetic BOLD from known ground-truth parameters
- **Solver:** attempts to recover those parameters from the synthetic BOLD

If parameters cannot be recovered in simulation, they cannot be interpreted in real data.

---

### 4.2 General Procedure

The following steps apply to every simulation, regardless of which model is being tested.

**Step A — Noiseless recovery check**

Before running any large-scale simulation, run the synthesizer → solver loop for a small set of synthetic voxels (~10–20) with **no noise added**. If parameters do not recover with near-perfect accuracy (>99%, following Kim et al.), this indicates a structural identifiability problem. No amount of data or regularisation can fix a structural identifiability failure — the model or the stimulus design must be reconsidered.

This check must be done across a range of parameter values, not just a single point, because identifiability can be locally valid but fail in other regions of parameter space.

**Step B — Generate synthetic voxel population**

Generate ~300 synthetic voxels. For each voxel, independently sample all ground-truth parameters from their plausible ranges. **Critically, all parameters must be sampled independently of each other.** If parameters are correlated during generation (e.g., because τ is initialised from CF, and CF is also sampled), correlations in the recovery errors cannot be detected as coupling.

Apply realistic subcortical fMRI noise to the synthesized timecourses. The noise level should be calibrated to match expected subcortical SNR, which is expected to be lower than the cortical benchmark used by Kim et al. (0.1 dB).

**Step C — Run the solver**

Fit the model to each synthetic BOLD timecourse using gradient descent with at least 3 random restarts. Record the best-fitting parameter set (lowest loss) for each voxel.

**Step D — Basic recovery metrics**

For each free parameter, compute:
- Scatter plot of estimated vs. ground-truth (identity line = perfect recovery)
- MAPE: median absolute percentage error across 300 voxels
- Pearson r between estimated and ground-truth values

These are the primary outputs of the simulation, directly analogous to Kim et al. Figure 4C–E.

**Step E — Parameter coupling check**

For each pair of parameters, test whether the recovery error in one parameter is predicted by the ground-truth value of another. See Section 4.7 for the full coupling diagnostic procedure, which is specific to Model 3.

---

### 4.3 Simulation 1 — Model 1: Spectral Only

**Purpose:** Verify that CF and α can be recovered from the BOLD responses to frequency sweeps. This is the prerequisite for all subsequent simulations — if CF is not recoverable, nothing downstream is valid.

**Ground-truth parameter sampling:**

| Parameter | Distribution | Range |
|---|---|---|
| CF | Log-uniform (Greenwood-spaced) | 125–4000 Hz |
| α | Uniform | TBD |

**Expected outcome:** CF recovery should be accurate (low MAPE, high Pearson r). α may be noisier. If CF MAPE exceeds ~15%, the frequency step design or number of runs is insufficient.

**Specific checks:**
- Does CF recovery degrade at frequency extremes (125 Hz, 4000 Hz)?
- Is α recovery biased in any CF region?
- Do CF and α couple (does error in α correlate with ground-truth CF)?

---

### 4.4 Simulation 2 — Model 2: Spectral + Duration

**Purpose:** Verify that `pref_dur` and `σ_dur` are recoverable from the 9 temporal conditions, independently of CF and α. This is the cleanest test of whether duration information in the stimulus design is sufficient to constrain duration tuning parameters, because there is no AdapTrans and therefore no τ to couple with.

**Ground-truth parameter sampling:**

| Parameter | Distribution | Range |
|---|---|---|
| CF | Log-uniform | 125–4000 Hz |
| α | Uniform | TBD |
| pref_dur | Uniform | 54–5000 ms |
| σ_dur | Uniform | 50–2000 ms |

All four parameters sampled independently.

**Expected outcome:** CF and α should recover as in Simulation 1. `pref_dur` and `σ_dur` should also recover, with MAPE that may be somewhat higher than CF given the fewer conditions constraining them.

**Specific checks:**
- Does `pref_dur` recovery degrade at extreme values (very short: ~54 ms, very long: ~5000 ms)?
- Does `σ_dur` couple with `pref_dur` (do wide and narrow tuning produce different biases)?
- Is recovery uniform across the range of durations tested by your 9 conditions?

If `pref_dur` does not recover here, the problem lies in the stimulus design, not model complexity. The remedy is to redesign the temporal conditions before proceeding to Models 3 and 4.

---

### 4.5 Simulation 3 — Model 4: Spectral + AdapTrans

**Purpose:** Verify that `τ_ON`, `τ_OFF`, `w`, ON weight, and OFF weight are recoverable without a Duration Gaussian. This establishes whether AdapTrans parameters are identifiable at all from the BOLD signal before asking whether they can be separated from an additional Gaussian.

**Ground-truth parameter sampling:**

| Parameter | Distribution | Range |
|---|---|---|
| CF | Log-uniform | 125–4000 Hz |
| α | Uniform | TBD |
| w | Uniform | 0–1 |
| ON weight | Uniform | TBD |
| OFF weight | Uniform | TBD |
| τ_ON | Uniform | 10–500 ms |
| τ_OFF | Uniform | 10–500 ms |

All parameters sampled independently. Note that `τ_ON` and `τ_OFF` are sampled independently of each other and of CF — do not initialise them from the Rançon equation during generation.

**Expected outcome:** τ_ON and τ_OFF are the hardest parameters to recover because they operate at sub-TR timescales. Some coupling between τ_ON and τ_OFF may be expected — a slow ON with a fast OFF may produce BOLD similar to a fast ON with a slow OFF under some conditions.

**Specific checks:**
- Does τ_ON couple with τ_OFF?
- Does w couple with ON/OFF weights?
- Does MAP prior tightness on τ affect recovery of other parameters?
- Vary σ_τ (the MAP prior width) and observe the trade-off between τ recovery accuracy and freedom from prior bias.

---

### 4.6 Simulation 4 — Model 3: Spectral + Duration + AdapTrans

**Purpose:** Test whether all 9 per-voxel parameters are jointly recoverable, and specifically diagnose coupling between `pref_dur` and `τ_ON/τ_OFF`. This is the most critical simulation and should only be run after Simulations 1–3 have passed.

**Ground-truth parameter sampling:**

| Parameter | Distribution | Range |
|---|---|---|
| CF | Log-uniform | 125–4000 Hz |
| α | Uniform | TBD |
| pref_dur | Uniform | 54–5000 ms |
| σ_dur | Uniform | 50–2000 ms |
| w | Uniform | 0–1 |
| ON weight | Uniform | TBD |
| OFF weight | Uniform | TBD |
| τ_ON | Uniform | 10–500 ms |
| τ_OFF | Uniform | 10–500 ms |

**All 9 parameters sampled independently.**

**Expected concern:** `pref_dur` and `τ_ON/τ_OFF` both produce duration-dependent BOLD response profiles. The Gaussian modulates amplitude as a function of tone duration; AdapTrans modulates amplitude via the degree of ON/OFF overlap, which also depends on tone duration relative to τ. The optimizer may trade one off against the other, producing a biased recovery. See Section 4.7 for the full coupling diagnostic.

---

### 4.7 The Coupling Diagnostic (Model 3 only)

This diagnostic is specific to Model 3 and addresses the key identifiability concern: whether `pref_dur` and `τ_ON/τ_OFF` can be independently recovered.

#### Why coupling is expected

For a voxel with a slow AdapTrans filter (large τ), the ON response does not fully decay before the OFF response fires, even for relatively long tones. This means the AdapTrans filter itself creates a preference for longer durations — because only long tones allow the ON and OFF responses to separate fully. This implicit duration preference functionally overlaps with what `pref_dur` is explicitly modelling.

The optimizer therefore has two ways to fit "this voxel prefers long durations": set `pref_dur` to a large value, or set τ to a large value. It may find any mixture of the two.

#### Step 1 — Targeted 2D grid test (run first, before 300-voxel simulation)

Fix all parameters to reasonable values except `pref_dur` and `τ_ON`. Generate a 5×5 grid of (pref_dur, τ_ON) combinations spanning their full ranges — 25 synthetic voxels total. Synthesise BOLD with noise and fit Model 3 to each.

Plot the recovered (pref_dur, τ_ON) pairs against the ground-truth grid. If the recovered values form a diagonal ridge — different ground-truth combinations all mapping to similar estimated values — the coupling is severe. If they cluster near the identity, parameters are separable.

This test is fast and immediately reveals the structure of the coupling before committing to a full 300-voxel simulation.

#### Step 2 — Binned scatter analysis

After running the full 300-voxel simulation, divide voxels into three equal bins by ground-truth τ_ON: low (10–170 ms), medium (170–340 ms), high (340–500 ms).

Within each bin, plot estimated `pref_dur` vs. ground-truth `pref_dur`. If coupling exists, the pattern will be:
- **High-τ bin:** estimated `pref_dur` is systematically higher than ground-truth (τ already "explains" duration preference; Gaussian is pushed toward longer durations)
- **Low-τ bin:** estimated `pref_dur` is less biased

Repeat with bins defined by ground-truth `pref_dur`, examining τ_ON recovery within each bin.

#### Step 3 — Regression test

For each voxel, compute the recovery error:

```
error_pref_dur = estimated_pref_dur - gt_pref_dur
```

Regress this error against all other ground-truth parameters:

```
error_pref_dur ~ gt_τ_ON + gt_τ_OFF + gt_w + gt_σ_dur + gt_CF + ...
```

A significant coefficient on `gt_τ_ON` or `gt_τ_OFF` confirms that recovery error in `pref_dur` is predicted by the true value of τ — the definition of coupling. Report F-statistics and p-values, following Kim et al.'s approach.

Repeat with `error_τ_ON` as the outcome and `gt_pref_dur` as a predictor. If both regressions are significant, the coupling is bidirectional (worst case).

#### Step 4 — Vary MAP prior width σ_τ

Re-run the 300-voxel simulation at several values of σ_τ (e.g., 25, 50, 100, 200 ms). For each, compute:
- MAPE for `pref_dur`
- MAPE for `τ_ON`
- Regression coefficient for coupling between them

This reveals the trade-off: tighter σ_τ reduces coupling at the cost of constraining τ to the prior. Choose the smallest σ_τ that brings `pref_dur` coupling below the severity threshold.

---

### 4.8 Model Misspecification Tests

In addition to per-model recovery, test what happens when the wrong model is fitted to data generated by another model. These tests assess how discriminable the models are in practice.

| Data generated by | Model fitted | Question |
|---|---|---|
| Model 1 | Model 2 | Does adding a Gaussian to a purely spectral voxel produce spurious duration estimates? |
| Model 2 | Model 4 | Can AdapTrans alone fit data generated by an explicit Gaussian? |
| Model 4 | Model 2 | Can an explicit Gaussian fit data generated by AdapTrans dynamics? |
| Model 2 | Model 3 | Does adding AdapTrans parameters to Gaussian-generated data produce coupling artefacts? |

If Model 4 fits Model 2 data well (high R²), the two models are not discriminable from the BOLD signal — your model comparison lacks power. If Model 4 fits Model 2 data poorly, the models make different predictions and can be distinguished.

---

## 5. Severity Thresholds and Remedies

### Recovery quality thresholds

| MAPE | Interpretation |
|---|---|
| < 15% | Acceptable recovery — proceed |
| 15–30% | Marginal — examine whether noise or coupling is the cause |
| > 30% | Problematic — do not proceed to real data |

### Coupling severity thresholds

| Coupling regression result | Interpretation | Remedy |
|---|---|---|
| Not significant, no bias in binned scatter | Parameters separable | Proceed |
| Significant but weak (small coefficient) | Mild coupling — manageable | Tighten σ_τ |
| Significant and strong, clear bias in binned scatter | Serious coupling | Apply Remedy 2 or 3 below |
| Structural failure in noiseless recovery | Model unidentifiable | Redesign required |

### Remedies in order of invasiveness

**Remedy 1 — Tighten MAP prior on τ**
Reduce σ_τ until coupling drops below threshold. The simulation directly tells you the required value. This is the least invasive option and should be tried first.

**Remedy 2 — Staged fitting**
In Stage 2 of real-data fitting, fit `pref_dur` and `σ_dur` with τ fixed at the Rançon initialisation. Only in Stage 3 jointly refine τ. If Stage 3 joint refinement destabilises `pref_dur`, report Stage 2 estimates as primary and Stage 3 as exploratory.

**Remedy 3 — Stimulus redesign**
The conditions that best separate `pref_dur` from τ are those at duration extremes, particularly the 5000 ms sustained condition (Condition 9). This condition is critical: for a voxel with large τ, the AdapTrans response to a 5000 ms tone is purely sustained (onset transient has decayed); for a voxel with a Gaussian peaked at 5000 ms, the response is large for a different reason. If coupling is severe, consider adding more conditions at very short and very long durations to put more leverage on the pref_dur/τ distinction.

---

## 6. Recommended Execution Order

Run simulations in this order. Stop and address failures before proceeding.

```
Step 1:  Noiseless recovery — all 4 models
         → Confirms no structural identifiability failures

Step 2:  Simulation 1 (Model 1) — Spectral only
         → Establishes CF recovery as the baseline
         → If CF fails here, fix before anything else

Step 3:  Simulation 2 (Model 2) — Spectral + Duration
         → Establishes pref_dur recovery without coupling risk
         → If pref_dur fails here, the stimulus design is insufficient

Step 4:  Simulation 3 (Model 4) — Spectral + AdapTrans
         → Establishes τ and w recovery without coupling risk from Gaussian
         → Determines appropriate σ_τ range for MAP prior

Step 5:  Targeted 2D grid test — Model 3 only
         → Fast pre-check of pref_dur / τ coupling geometry
         → Run before committing to full 300-voxel Model 3 simulation

Step 6:  Simulation 4 (Model 3) — Full coupling diagnostic
         → Full 300-voxel simulation with coupling regression
         → Apply remedies if needed; rerun until passing

Step 7:  Model misspecification tests
         → Assess discriminability between models before real data collection
```

Only after all simulations pass do experimental data collection and real-data fitting proceed.

---

*Document based on Kim et al. (2024) spatiotemporal pRF framework, Rançon et al. (2024) AdapTrans model, and project design decisions as of April 2026.*
