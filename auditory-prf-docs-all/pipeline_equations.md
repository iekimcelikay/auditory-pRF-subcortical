# Auditory pRF Pipeline — Mathematical Formulation

This document describes the mathematical equations implemented in `full_pipeline_with_adaptrans.py`.

## Notation

| Symbol | Description | Code variable |
|--------|-------------|---------------|
| $R(k, t)$ | Population PSTH: firing rate at CF index $k$, time bin $t$ | `population_psth[k, t]` |
| $\alpha$ | Power-law sharpening exponent | `alpha` |
| $k_0$ | Target CF index | `cf_index` |
| $f_{k_0}$ | Characteristic frequency of target CF (Hz) | `cf_hz` |
| $d$ | Tone duration (ms) | `tone_dur_ms` |
| $\hat{d}$ | Preferred duration (ms) | `pref_dur` |
| $\sigma_d$ | Duration tuning width (ms) | `sigma_dur` |
| $w$ | Adaptation weight (kernel shape) | `w` |
| $\tau(f)$ | CF-dependent time constant (ms) | `tau_ms` |
| $a$ | Exponential decay rate | `a` |
| $K$ | FIR kernel length (samples) | `K` |
| $\rho$ | ON/OFF combination ratio | `on_off_ratio` |
| $S$ | Number of tones in the sequence | `n_tones` |

---

## Step 1–2: Load and extract CF timecourse

Load the population PSTH matrix $R(k, t)$ of shape $(N_\text{CFs}, N_\text{bins})$ from the cochlear simulation `.npz` file. Select the target CF row:

$$R_{k_0}(t) = R(k_0, t)$$

---

## Step 3: Power-law sharpening (lateral inhibition)

Apply element-wise power-law to the **full population** matrix, then rescale to preserve the grand mean:

$$\tilde{R}(k, t) = R(k, t)^\alpha \cdot \frac{\langle R \rangle}{\langle R^\alpha \rangle}$$

where $\langle \cdot \rangle$ denotes the mean over all CFs and all time bins:

$$\langle R \rangle = \frac{1}{N_\text{CFs} \cdot N_\text{bins}} \sum_{k,t} R(k,t)$$

Then extract the target CF row from the sharpened population:

$$\tilde{R}_{k_0}(t) = \tilde{R}(k_0, t)$$

> **Why rescale?** The power-law amplifies differences between high and low rates (sharpening frequency tuning), but changes the overall magnitude. Dividing by the post-sharpening mean and multiplying by the pre-sharpening mean keeps the grand mean firing rate unchanged, so downstream stages see comparable amplitudes regardless of $\alpha$.

---

## Step 4: Chunk into tone-ON windows

Parse tone timing from the stimulus filename (`dur<N>ms`, `isi<N>ms`). Compute onset/offset times for each tone $s = 1, \ldots, S$:

$$t_s^\text{on} = (s-1) \cdot (d + \Delta_\text{ISI})$$

$$t_s^\text{off} = t_s^\text{on} + d$$

Extract each tone-ON window (plus a 50 ms margin) from $\tilde{R}_{k_0}(t)$ and compute the mean firing rate per tone:

$$\bar{r}_s = \frac{1}{|\mathcal{W}_s|} \sum_{t \in \mathcal{W}_s} \tilde{R}_{k_0}(t)$$

where $\mathcal{W}_s = \{t : t_s^\text{on} \leq t < t_s^\text{off} + 50\text{ ms}\}$.

---

## Step 5: Duration Gaussian filter (scalar)

Weight each tone's mean rate by a Gaussian centered on the preferred duration:

$$g(d) = \frac{1}{\sqrt{2\pi}\,\sigma_d} \exp\!\left(-\frac{(d - \hat{d})^2}{2\sigma_d^2}\right)$$

$$p_s = \bar{r}_s \cdot g(d)$$

where $p_s$ is the **pRF response scalar** for tone $s$. Since all tones in a pure-tone sequence share the same duration $d$, $g(d)$ is the same scalar for every tone — it modulates the overall amplitude based on how well the tone duration matches the neuron's preferred duration.

---

## Step 6: Build boxcar impulse train

Construct a 1 ms resolution signal $x(n)$ where each tone's interval is filled with its pRF response amplitude. For each tone $s$, processed **in isolation**:

$$x_s(n) = \begin{cases} p_s & \text{if } n_{s}^\text{on} \leq n < n_{s}^\text{off} \\ 0 & \text{otherwise} \end{cases}$$

where $n_s^\text{on} = \text{round}(t_s^\text{on} / \Delta t)$ and $n_s^\text{off} = \text{round}(t_s^\text{off} / \Delta t)$ with $\Delta t = 1$ ms.

---

## Step 7: AdapTrans ON/OFF filters

### Time constant (Willmore et al., 2016, rescaled for subcortex)

$$\tau(f) = 0.15 \cdot \left(500 - 105 \cdot \log_{10}(f)\right) \quad \text{[ms]}$$

### Decay rate

$$a = e^{-\Delta t / \tau(f)}$$

### Kernel length (auto)

If not specified:

$$K = \lceil 3 \cdot \tau_\text{max} / \Delta t \rceil$$

### ON kernel (FIR, length $K$)

$$C = \frac{1}{\displaystyle\sum_{j=0}^{K-2} a^j}$$

$$h_\text{ON}[n] = \begin{cases} +1 & n = 0 \quad \text{(current sample)} \\ -C \cdot w \cdot a^{n-1} & n = 1, \ldots, K-1 \quad \text{(exponentially weighted past)} \end{cases}$$

The ON kernel computes: *current sample minus adapted running average of the past*. Large positive output = onset (increase from baseline).

### OFF kernel (FIR, length $K$)

The OFF kernel detects *decreases* — it compares the exponentially weighted past against the (discounted) current sample:

$$h_\text{OFF}[n] = \begin{cases} -w & n = 0 \quad \text{(current sample, discounted by } w\text{)} \\ +C \cdot a^{n-1} & n = 1, \ldots, K-1 \quad \text{(exponentially weighted past)} \end{cases}$$

### ON/OFF kernel asymmetry

The OFF kernel is **not** the exact negative of the ON kernel. This is intentional. Comparing the two side by side:

| Tap | $h_\text{ON}[n]$ | $h_\text{OFF}[n]$ |
|-----|-------------------|--------------------|
| $n = 0$ (present) | $+1$ | $-w$ |
| $n \geq 1$ (past) | $-C w a^{n-1}$ | $+C a^{n-1}$ |

The asymmetry: **ON discounts the past by $w$; OFF discounts the present by $w$.**

If $h_\text{OFF} = -h_\text{ON}$ were true, we would need tap 0 $= -1$ and taps $n \geq 1$ $= +Cwa^{n-1}$. Instead we have tap 0 $= -w$ and taps $n \geq 1$ $= +Ca^{n-1}$ (no $w$ factor). The two kernels are related, but not by simple negation.

Intuitively:
- **ON**: "Is the current sample ($\times 1$) larger than the adapted past ($\times w$)?"
- **OFF**: "Is the adapted past ($\times 1$) larger than the current sample ($\times w$)?"

When $w = 1$, the asymmetry vanishes and $h_\text{OFF} = -h_\text{ON}$ exactly.

### Code shortcut for OFF kernel derivation

The implementation derives the OFF kernel from the ON kernel algebraically, avoiding redundant computation of $C$ and the exponential terms:

**Step 1.** Negate and divide by $w$:

$$\frac{-h_\text{ON}[n]}{w} = \begin{cases} -1/w & n = 0 \\ +C a^{n-1} & n \geq 1 \end{cases}$$

For taps $n \geq 1$, the $w$ cancels cleanly:

$$\frac{-(-Cwa^{n-1})}{w} = +Ca^{n-1} \quad \checkmark$$

But for tap 0:

$$\frac{-(+1)}{w} = -\frac{1}{w} \neq -w$$

**Step 2.** Overwrite tap 0:

$$h_\text{OFF}[0] \leftarrow -w$$

This yields the correct OFF kernel. In code:

```python
off_kernel = -on_kernel / w      # fixes taps 1..K-1, but tap 0 = -1/w (wrong)
off_kernel[0] = -w               # overwrite tap 0 to correct value
```

### Convolution (causal, per-tone isolation)

Each isolated single-tone boxcar $x_s(n)$ is left-padded with zeros (length $K-1$) and convolved with the ON and OFF kernels:

$$y_s^\text{ON}(n) = \sum_{m=0}^{K-1} h_\text{ON}[m] \cdot x_s(n - m)$$

$$y_s^\text{OFF}(n) = \sum_{m=0}^{K-1} h_\text{OFF}[m] \cdot x_s(n - m)$$

### Superposition across tones

The per-tone responses are summed (exploiting the linearity of convolution):

$$Y^\text{ON}(n) = \sum_{s=1}^{S} y_s^\text{ON}(n)$$

$$Y^\text{OFF}(n) = \sum_{s=1}^{S} y_s^\text{OFF}(n)$$

---

## Step 8: ON/OFF combination

The final combined pRF timecourse uses a single ratio parameter:

$$Y(n) = \rho \cdot Y^\text{ON}(n) + Y^\text{OFF}(n)$$

where:
- $\rho > 1$: onset-dominated response
- $\rho = 1$: balanced ON/OFF
- $\rho < 1$: offset-dominated response

---

## Full model (compact form)

For a given parameter set $\Theta = \{k_0, \alpha, \hat{d}, \sigma_d, w, \rho\}$:

$$Y(n \mid \Theta) = \rho \sum_{s=1}^{S} \left(h_\text{ON} * x_s\right)(n) + \sum_{s=1}^{S} \left(h_\text{OFF} * x_s\right)(n)$$

where each $x_s$ encodes the pRF response scalar:

$$x_s(n) = \underbrace{\bar{r}_s}_{\text{mean rate}} \cdot \underbrace{g(d)}_{\text{duration weight}} \cdot \mathbf{1}_{[n_s^\text{on},\, n_s^\text{off})}(n)$$

and the kernels $h_\text{ON}, h_\text{OFF}$ depend on $f_{k_0}$ (via $\tau$) and $w$.
