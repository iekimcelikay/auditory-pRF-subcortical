"""Run design of the HARMONIC FM sweeps with silences (family P2, auditory-prf-docs-all/fm_sweep_silence_kim_design.md).

Every stimulus is a harmonic-complex sweep (f0 sweeps 125-800 Hz, components up to 2500 Hz,
SoundGen.harmonic_sweep_maker), arranged as tones separated by ISIs. This module only builds the run orders; the design
objects the recovery scripts use (name, label, folders, run orders) come from stimuli/sweep_designs.py.

One run = opening blank + (24 harmonic-sweep conditions + 8 nulls, shuffled) x 20 s + closing blank, with the timing
constants of the ERB-band design (TR 1.6 s, 4 TR blanks, no ITI, 25 % nulls), i.e. 652.8 s = 408 TRs. Every run holds all
24 conditions of its layout once, in its own random order (one order per design seed, 1000 ... 1023 for the 24-run
experiment). Null trials present the silent trial, whose cochlear response (silence_20s) is placed in the run like any
other trial.

Designs (stimuli/sweep_design_names.py):
    harmonic_sweep_armA   layout A: 3 rates x 2 tones x 4 ISIs
    harmonic_sweep_armB   layout B: 3 rates x 4 tones x 2 ISIs
A trial's soundfileid is the WAV stem written by stimuli/render_harmonic_sweep_wavs.py, e.g.
harmonic_sweep_armB_01_rate6hz_dur167ms_isi167ms, which is also the soundfileid of its combined cochlea result.

Usage
-----
    from stimuli.sweep_design_harmonic_complex import HarmonicSweepRunDesign
    design = HarmonicSweepRunDesign("A")
    run_design = design.build_run_design(1000)            # list of (soundfileid, onset_s)
    total_s = design.total_run_duration_s(run_design)     # 652.8
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from auditory_prf.prf_pipeline.run_assembly import generate_run_design
from stimuli.render_harmonic_sweep_wavs import layout_conditions, wav_stem
from stimuli.sweep_design_centered import (
    CLOSING_BLANK_S,
    ITI_RANGE_S,
    NULL_FRACTION,
    OPENING_BLANK_S,
    TR_S,
    TRIAL_DURATION_S,
)
from stimuli.sweep_design_names import HARMONIC_LAYOUT_COUNTS, harmonic_design_name

SILENCE_SEQ_ID = f"silence_{TRIAL_DURATION_S:.0f}s"


class HarmonicSweepRunDesign:
    """Run orders of one layout of the harmonic-sweep stimuli.

    Parameters
    ----------
    layout : {"A", "B"}
        A: 2 tones x 4 ISIs, B: 4 tones x 2 ISIs (x 3 rates = 24 conditions each).
    """

    TR_S = TR_S
    TRIAL_DURATION_S = TRIAL_DURATION_S
    SILENCE_SEQ_ID = SILENCE_SEQ_ID

    def __init__(self, layout: str):
        assert layout in HARMONIC_LAYOUT_COUNTS, layout
        self.layout = layout
        self.name = harmonic_design_name(layout)
        self.conditions = layout_conditions(layout)
        self._seq_id_by_index = {c["index"]: wav_stem(layout, c) for c in self.conditions}

    @property
    def seq_ids(self) -> list:
        """soundfileids of the 24 conditions, in condition-index order (silence not included)."""
        return [self._seq_id_by_index[c["index"]] for c in self.conditions]

    def seq_id(self, condition_index: int, unused_isi_ms: int, freq_hz) -> str:
        """soundfileid of one trial, in the (tone_on_ms, isi_ms, freq_hz) form generate_run_design expects.

        condition_index takes the tone_on_ms slot, unused_isi_ms is ignored (the signature is fixed by
        generate_run_design) and freq_hz is None for nulls, which map to SILENCE_SEQ_ID.
        """
        if freq_hz is None:
            return SILENCE_SEQ_ID
        return self._seq_id_by_index[condition_index]

    def build_base_trials(self, null_fraction: float = NULL_FRACTION) -> list:
        """Master trial list for ONE run: every condition once plus nulls.

        Returns
        -------
        list of (condition_index, 0, 1.0) for sweeps and (0, 0, None) for nulls.
        """
        sweep_trials = [(c["index"], 0, 1.0) for c in self.conditions]
        n_null = int(np.floor(len(sweep_trials) * null_fraction / (1 - null_fraction) + 1e-9))
        return sweep_trials + [(0, 0, None)] * n_null

    def build_run_design(self, seed: int) -> list:
        """Shuffled run design: list of (soundfileid, onset_s); nulls are SILENCE_SEQ_ID."""
        return generate_run_design(
            self.build_base_trials(),
            self.seq_id,
            trial_duration_s=TRIAL_DURATION_S,
            opening_blank_s=OPENING_BLANK_S,
            iti_range_s=ITI_RANGE_S,
            seed=seed,
        )

    @staticmethod
    def total_run_duration_s(run_design: list) -> float:
        """Run length implied by a design: last onset + trial + closing blank."""
        return run_design[-1][1] + TRIAL_DURATION_S + CLOSING_BLANK_S
