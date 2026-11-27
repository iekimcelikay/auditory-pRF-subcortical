"""Render the tau-probe stimuli (static harmonic complex tones separated by ISIs) to mono 16-bit .wav files.

Purpose: probe the AdapTrans time constant tau with tone/ISI conditions after the nine-condition temporal grid of Kim et
al. 2024 (conditions 1-8, values changed for tau). Every stimulus is a STATIC harmonic complex (fixed f0, no sweep). A
trial is exactly 20 s of identical [tone, ISI] blocks and ends with silence. Tone and ISI are both <= 1000 ms and
tone + ISI = 20 s / n_blocks exactly (n_blocks integer, so the blocks tile the trial in whole samples).

Kim grid, with x the fixed value and the periods p1, p2, p3 = 20 s / (N1, N2, N3), o_k = p_k - x:
    conditions 1-3   tone x,    ISI o_k          (ISI varies at a fixed tone)
    conditions 4-6   tone o_k,  ISI x            (the mirror: tone varies at a fixed ISI)
    conditions 7-8   condition 1 scaled by s2, s3 (same tone/ISI ratio, N1 / s blocks)
Condition 9 of Kim et al. (one stimulus filling the window) is not used (a tone longer than 1000 ms is not allowed).

    harmonic_tau_setA   x = 100 ms, N = (120, 48, 20), s = (8, 10)   (chosen; auditory-prf-docs-all/tau_design_candidate_set_A_x100.md)
    harmonic_tau_setB   x = 200 ms, N = (48, 40, 20),  s = (3, 4)    (kept as a future candidate)

The 8 conditions are repeated 3 times in a run, each repetition with a DIFFERENT f0 (default 100, 120, 170 Hz), so a
trial is a (repetition = f0, Kim condition) pair: 24 trials per run. With a normal (resolved-harmonic) complex a single
f0 drives only the CFs near its harmonics; three f0 values together drive almost every one of the model's 30 channels
(125-2500 Hz): rough estimate, summed over the three f0 the weakest channel gets 0.38x the drive of the best, against
0.07x for the best single f0 (a 2500 Hz harmonic ceiling would leave the 2300 and 2500 Hz channels at 0.14x and 0.03x).
Dense low-f0 complexes (40-50 Hz) were rejected by listening (rough, buzzing); sweeps are not used.

Carrier (all of it is an argument): harmonics n * f0 up to --max-harmonic-hz (default 3000 Hz; linear fade over the
--taper-hz below it, default 500 Hz),
amplitude proportional to ERB(f)^-1/2 (equal energy per ERB), Schroeder phases by default. The tone is NOT cut to a whole
number of f0 periods (the 10 ms sine ramps hide the cut), so the exact tone and ISI lengths of the design are kept. Every
tone has the same level: its RMS, ramps included, is the 60 dB SPL RMS of SoundGen (the same rule as
stimuli/render_harmonic_sweep_wavs.py).

One folder per design, named like the design: <out-root>/<design name>/ holds
    <design name>_<index>_f0<hz>hz_dur<ms>ms_isi<ms>ms.wav    index 01-24: f0 (repetition) first, then Kim condition
    silence_20s.wav     the null trial; sorts after the tone files: the 25th WAV, the last index of the cochlea job
    design_info.json    the design, rendering parameters, estimated CF drive and the 24 conditions (exact samples)
The cochlea runner (examples/cochlea_sweep_wavs.py) and the design registry (stimuli/sweep_design_names.py) do not know
this design name yet.

Usage (from the project root)
-----------------------------
    python stimuli/render_harmonic_tau_wavs.py                          # set A, f0 100 120 170 -> stimuli/produced/harmonic_tau_setA/
    python stimuli/render_harmonic_tau_wavs.py --set B --f0-hz 110 140 180
    python stimuli/render_harmonic_tau_wavs.py --out-root /scratch/.../stimuli/produced
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from stimuli.render_harmonic_sweep_wavs import (
    DBSPL,
    PHASES_DEFAULT,
    RAMP_S,
    SAMPLE_RATE_HZ,
    TAPER_HZ_DEFAULT,
    TRIAL_DURATION_S,
    TRIAL_SAMPLES,
    reference_rms,
    render_silence_wav,
    write_mono_wav,
)
from stimuli.soundgen import SoundGen

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parents[1]))

# ── Design constants ──────────────────────────────────────────────────────────
DESIGN_PREFIX = "harmonic_tau_set"
SETS = {
    "A": {"x_ms": 100.0, "n_blocks": (120, 48, 20), "scales": (8, 10)},
    "B": {"x_ms": 200.0, "n_blocks": (48, 40, 20), "scales": (3, 4)},
}
MAX_TONE_ISI_MS = 1000.0
MIN_TONE_SAMPLES = int(round(2 * RAMP_S * SAMPLE_RATE_HZ))         # a tone must hold both 10 ms ramps
N_KIM_CONDITIONS = 8
F0S_HZ_DEFAULT = (100.0, 120.0, 170.0)                              # one per repetition (user: the triple seems fine)
MAX_HARMONIC_HZ_DEFAULT = 3000.0       # above the 2500 Hz top channel, so it is driven (2500 Hz left it at 0.03x; user: 3000 sounds acceptable)
CF_RANGE_HZ = (125.0, 2500.0)                                       # the 30 channels of the cochlear model (examples/cochlea_sweep_wavs.py)
N_CF_FOR_DRIVE = 30


def design_name(set_letter: str) -> str:
    """'A' -> 'harmonic_tau_setA'."""
    assert set_letter in SETS, set_letter
    return f"{DESIGN_PREFIX}{set_letter}"


def design_label(set_letter: str) -> str:
    return f"Harmonic tones, Kim-grid set {set_letter} x 3 f0 (tone/ISI, tau probe)"


def erb_bandwidth_hz(freq_hz) -> np.ndarray:
    """Equivalent rectangular bandwidth in Hz (Glasberg & Moore 1990); array in, array out."""
    return 24.7 * (4.37 * np.asarray(freq_hz, dtype=float) / 1000.0 + 1.0)


# ── Conditions ────────────────────────────────────────────────────────────────
def set_conditions(set_letter: str) -> list:
    """The 8 timing conditions (Kim conditions 1-8) of a set, in whole samples.

    Returns
    -------
    list of dict, keys: index (1-8, Kim condition), n_tone_samples, n_isi_samples, n_blocks, tone_ms, isi_ms.
    """
    spec = SETS[set_letter]
    n_x = int(round(spec["x_ms"] * SAMPLE_RATE_HZ / 1000.0))
    assert abs(n_x - spec["x_ms"] * SAMPLE_RATE_HZ / 1000.0) < 1e-9, "x must be a whole number of samples"
    n_blocks_list = spec["n_blocks"]
    for n_blocks in n_blocks_list:
        assert TRIAL_SAMPLES % n_blocks == 0, f"{n_blocks} blocks do not tile the trial in whole samples"
    periods = [TRIAL_SAMPLES // n_blocks for n_blocks in n_blocks_list]
    partners = [period - n_x for period in periods]                  # o_k in samples
    pairs = [(n_x, partner, n_blocks) for partner, n_blocks in zip(partners, n_blocks_list)]       # conditions 1-3
    pairs += [(partner, n_x, n_blocks) for partner, n_blocks in zip(partners, n_blocks_list)]      # conditions 4-6
    for scale in spec["scales"]:                                                                    # conditions 7-8
        assert n_blocks_list[0] % scale == 0, f"scale {scale} does not divide {n_blocks_list[0]} blocks"
        pairs.append((scale * n_x, scale * partners[0], n_blocks_list[0] // scale))
    conditions = []
    for index, (n_tone, n_isi, n_blocks) in enumerate(pairs, start=1):
        assert n_blocks * (n_tone + n_isi) == TRIAL_SAMPLES, f"condition {index} does not tile the trial"
        assert MIN_TONE_SAMPLES <= n_tone <= MAX_TONE_ISI_MS * SAMPLE_RATE_HZ / 1000.0, f"condition {index}: tone"
        assert 0 < n_isi <= MAX_TONE_ISI_MS * SAMPLE_RATE_HZ / 1000.0, f"condition {index}: ISI"
        conditions.append({"index": index, "n_tone_samples": int(n_tone), "n_isi_samples": int(n_isi),
                           "n_blocks": int(n_blocks), "tone_ms": 1000.0 * n_tone / SAMPLE_RATE_HZ,
                           "isi_ms": 1000.0 * n_isi / SAMPLE_RATE_HZ})
    assert len(conditions) == N_KIM_CONDITIONS
    assert len({(c["n_tone_samples"], c["n_isi_samples"]) for c in conditions}) == len(conditions), "duplicate condition"
    return conditions


def trial_conditions(set_letter: str, f0s_hz: Sequence[float] = F0S_HZ_DEFAULT) -> list:
    """All trials of a run: every Kim condition once per f0 (= repetition), f0 first, then Kim condition.

    Returns
    -------
    list of dict, the keys of set_conditions plus: index (1-based over all trials), kim_condition (1-8),
    repetition (1-based, one per f0), f0_hz.
    """
    f0s_hz = [float(f0) for f0 in f0s_hz]
    assert len(f0s_hz) >= 1 and len(set(f0s_hz)) == len(f0s_hz), "one distinct f0 per repetition"
    kim = set_conditions(set_letter)
    trials = []
    for repetition, f0_hz in enumerate(f0s_hz, start=1):
        for condition in kim:
            trials.append({**condition, "index": len(trials) + 1, "kim_condition": condition["index"],
                           "repetition": repetition, "f0_hz": f0_hz})
    return trials


def wav_stem(set_letter: str, trial: dict) -> str:
    """File name without extension, e.g. harmonic_tau_setA_01_f0100hz_dur100ms_isi67ms (ms rounded)."""
    return (f"{design_name(set_letter)}_{trial['index']:02d}_f0{trial['f0_hz']:g}hz"
            f"_dur{round(trial['tone_ms'])}ms_isi{round(trial['isi_ms'])}ms")


# ── Carrier ───────────────────────────────────────────────────────────────────
def harmonic_amplitudes(
    f0_hz: float,
    max_harmonic_hz: float = MAX_HARMONIC_HZ_DEFAULT,
    taper_hz: float = TAPER_HZ_DEFAULT,
) -> tuple:
    """Frequencies and relative amplitudes of the harmonics: equal energy per ERB, linear fade at the top.

    Amplitude of harmonic n is ERB(n * f0) ** -0.5 (so amplitude ** 2 * ERB / f0, the energy per ERB, is the same for
    every harmonic) times the fade weight clip((max_harmonic_hz - n * f0) / taper_hz, 0, 1) (hard cutoff if taper_hz = 0).

    Returns
    -------
    harmonic_hz : np.ndarray, shape (n_harmonics,)
    amplitude : np.ndarray, shape (n_harmonics,), arbitrary overall scale (the tone is calibrated afterwards)
    """
    assert f0_hz > 0 and taper_hz >= 0 and max_harmonic_hz < SAMPLE_RATE_HZ / 2
    n_harmonics = int(np.floor(max_harmonic_hz / f0_hz))
    assert n_harmonics >= 1, "f0 above max_harmonic_hz"
    harmonic_hz = f0_hz * np.arange(1, n_harmonics + 1)
    if taper_hz > 0:
        fade = np.clip((max_harmonic_hz - harmonic_hz) / taper_hz, 0.0, 1.0)
    else:
        fade = (harmonic_hz <= max_harmonic_hz).astype(float)
    return harmonic_hz, fade * erb_bandwidth_hz(harmonic_hz) ** -0.5


def estimated_cf_drive(harmonic_hz: np.ndarray, amplitude: np.ndarray, cf_range_hz: tuple = CF_RANGE_HZ,
                       n_cf: int = N_CF_FOR_DRIVE) -> tuple:
    """Rough drive of each CF: power of the harmonics seen through a 4th-order gammatone magnitude response.

    |H(f)|^2 = (1 + ((f - cf) / b) ** 2) ** -4 with b = 1.019 ERB(cf) (low-level bandwidth). A diagnostic only: no
    cochlear model, no compression, no level-dependent broadening, so off-harmonic CFs probably respond more than this.

    Returns
    -------
    cf_hz : np.ndarray, shape (n_cf,), log-spaced over cf_range_hz
    drive : np.ndarray, shape (n_cf,), relative power
    """
    cf_hz = np.geomspace(cf_range_hz[0], cf_range_hz[1], n_cf)
    bandwidth_hz = 1.019 * erb_bandwidth_hz(cf_hz)
    gain_squared = (1.0 + ((harmonic_hz[None, :] - cf_hz[:, None]) / bandwidth_hz[:, None]) ** 2) ** -4   # (n_cf, n_harm)
    return cf_hz, (gain_squared * amplitude[None, :] ** 2).sum(axis=1)


def render_tone(
    n_tone_samples: int,
    f0_hz: float,
    max_harmonic_hz: float = MAX_HARMONIC_HZ_DEFAULT,
    taper_hz: float = TAPER_HZ_DEFAULT,
    phases: str = PHASES_DEFAULT,
) -> np.ndarray:
    """One static harmonic-complex tone with 10 ms sine ramps and RMS (ramps included) equal to the DBSPL RMS.

    Returns
    -------
    np.ndarray, shape (n_tone_samples,), file units (|x| < 1)
    """
    assert phases in ("schroeder", "sine"), phases
    harmonic_hz, amplitude = harmonic_amplitudes(f0_hz, max_harmonic_hz, taper_hz)
    harmonic_number = np.arange(1, len(harmonic_hz) + 1)
    if phases == "schroeder":
        phase_rad = -np.pi * harmonic_number * (harmonic_number - 1) / len(harmonic_number)
    else:
        phase_rad = np.zeros(len(harmonic_number))
    time_s = np.arange(n_tone_samples) / SAMPLE_RATE_HZ
    tone = np.zeros(n_tone_samples)
    for freq_hz, amp, phase in zip(harmonic_hz, amplitude, phase_rad):
        if amp > 0:
            tone += amp * np.sin(2 * np.pi * freq_hz * time_s + phase)
    tone = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S).sine_ramp(tone)
    assert tone.shape == (n_tone_samples,)
    return tone * (reference_rms() / np.sqrt(np.mean(tone ** 2)))      # tone as played (ramps included) = target RMS


def render_trial(trial: dict, **carrier_kwargs) -> np.ndarray:
    """One 20 s trial: n identical [tone, ISI] blocks at the trial's f0, ending with silence.

    Returns
    -------
    np.ndarray, shape (TRIAL_SAMPLES,), file units
    """
    tone = render_tone(trial["n_tone_samples"], trial["f0_hz"], **carrier_kwargs)
    block = np.concatenate([tone, np.zeros(trial["n_isi_samples"])])
    sound = np.tile(block, trial["n_blocks"])
    assert sound.shape == (TRIAL_SAMPLES,) and sound[-1] == 0.0
    assert np.abs(sound).max() < 1.0, "trial would clip"
    return sound


def write_design_info(set_letter: str, out_dir: Path, f0s_hz: Sequence[float], carrier_params: dict) -> Path:
    """design_info.json: design name and label, rendering parameters, estimated CF drive and the trials."""
    drive_by_f0 = {}
    for f0_hz in f0s_hz:
        _, drive = estimated_cf_drive(*harmonic_amplitudes(f0_hz, carrier_params["max_harmonic_hz"],
                                                           carrier_params["taper_hz"]))
        drive_by_f0[f0_hz] = drive
    summed = sum(drive_by_f0.values())
    spec = SETS[set_letter]
    info = {
        "design": design_name(set_letter),
        "label": design_label(set_letter),
        "set": set_letter,
        "structure": "Kim et al. 2024 temporal grid, conditions 1-8: 1-3 fixed tone, 4-6 fixed ISI, 7-8 scaled copies of 1",
        "x_ms": spec["x_ms"], "n_blocks_cond_1_to_3": list(spec["n_blocks"]), "scales_cond_7_8": list(spec["scales"]),
        "repetitions_per_run": len(f0s_hz), "f0s_hz": [float(f0) for f0 in f0s_hz],
        "carrier": "static harmonic complex, equal energy per ERB, tone not cut to whole f0 periods, one f0 per repetition",
        "sample_rate_hz": SAMPLE_RATE_HZ, "trial_duration_s": TRIAL_DURATION_S, "dbspl": DBSPL, "ramp_s": RAMP_S,
        "max_tone_isi_ms": MAX_TONE_ISI_MS,
        **carrier_params,
        "estimated_cf_drive_min_over_max_single_f0": {f"{f0:g}": float(d.min() / d.max()) for f0, d in drive_by_f0.items()},
        "estimated_cf_drive_min_over_max_summed": float(summed.min() / summed.max()),
        "conditions": [{**trial, "wav": f"{wav_stem(set_letter, trial)}.wav"}
                       for trial in trial_conditions(set_letter, f0s_hz)],
    }
    info_path = out_dir / "design_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    return info_path


def render_set(
    set_letter: str,
    out_dir: Path,
    f0s_hz: Sequence[float] = F0S_HZ_DEFAULT,
    write_silence: bool = True,
    max_harmonic_hz: float = MAX_HARMONIC_HZ_DEFAULT,
    taper_hz: float = TAPER_HZ_DEFAULT,
    phases: str = PHASES_DEFAULT,
) -> list:
    """Render the trials of one set (8 conditions x len(f0s_hz)) into out_dir (+ silence_20s.wav and design_info.json).

    Returns
    -------
    list of Path, the written .wav files in sorted (= cochlea array-index) order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    carrier_params = {"max_harmonic_hz": max_harmonic_hz, "taper_hz": taper_hz, "phases": phases}
    trials = trial_conditions(set_letter, f0s_hz)
    written = []
    for trial in trials:
        wav_path = out_dir / f"{wav_stem(set_letter, trial)}.wav"
        write_mono_wav(wav_path, render_trial(trial, **carrier_params))
        written.append(wav_path)
        print(f"{trial['index']}/{len(trials)} {wav_path.name}", flush=True)
    if write_silence:
        written.append(render_silence_wav(out_dir))
    write_design_info(set_letter, out_dir, f0s_hz, carrier_params)
    return written


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--set", choices=tuple(SETS), default="A", help="Kim-grid value set (default: A, the chosen one).")
    parser.add_argument("--out-root", type=str, default=str(PROJECT_ROOT / "stimuli" / "produced"),
                        help="Folder that receives one sub-folder per design, named like the design.")
    parser.add_argument("--f0-hz", type=float, nargs="+", default=list(F0S_HZ_DEFAULT),
                        help="One fundamental per repetition of the 8 conditions (default 100 120 170 Hz).")
    parser.add_argument("--max-harmonic-hz", type=float, default=MAX_HARMONIC_HZ_DEFAULT)
    parser.add_argument("--taper-hz", type=float, default=TAPER_HZ_DEFAULT)
    parser.add_argument("--phases", choices=("schroeder", "sine"), default=PHASES_DEFAULT,
                        help="Component phases: schroeder (smooth, default) or all in sine phase (pulse-like).")
    parser.add_argument("--no-silence", action="store_true", help="Do not write silence_20s.wav.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_root) / design_name(args.set)
    print(f"== {design_name(args.set)}: {design_label(args.set)} f0 {args.f0_hz} -> {out_dir}", flush=True)
    written = render_set(args.set, out_dir, f0s_hz=args.f0_hz, write_silence=not args.no_silence,
                         max_harmonic_hz=args.max_harmonic_hz, taper_hz=args.taper_hz, phases=args.phases)
    print(f"{len(written)} wav files in {out_dir}")


if __name__ == "__main__":
    main()
