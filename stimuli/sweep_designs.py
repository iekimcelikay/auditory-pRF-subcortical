"""The sweep designs by name: what the recovery scripts get from ``--design``.

    from stimuli.sweep_designs import get_design
    design = get_design("harmonic_sweep_armA")
    design.label                       # 'Harmonic sweeps: 3 rates x 2 tones x 4 ISIs'  (shown in plots)
    design.combined_dir                # 'models_output/harmonic_sweep_armA/combined'
    run_design = design.build_run_design(1000)
    design.total_run_duration_s(run_design)

Names, labels and folder rules are in stimuli/sweep_design_names.py (everything of a design lives under its name); the run
orders come from stimuli/sweep_design_erb.py and stimuli/sweep_design_harmonic_complex.py. Every design object has the
same attributes, so the recovery scripts do not care which one they got.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parents[1]))

from stimuli import sweep_design_names as names
from stimuli.sweep_design_names import DEFAULT_DESIGN_NAME, DESIGN_NAMES, ERB_DESIGN_NAME

__all__ = ["SweepDesign", "get_design", "DESIGN_NAMES", "DEFAULT_DESIGN_NAME"]


@dataclass(frozen=True)
class SweepDesign:
    """One sweep design: identity, run orders and the folders of its files (project-root-relative)."""
    name: str
    label: str
    build_run_design: Callable          # (seed) -> list of (soundfileid, onset_s)
    total_run_duration_s: Callable      # (run_design) -> seconds
    SILENCE_SEQ_ID: str
    TR_S: float
    wav_dir: str
    cochlea_dir: str
    combined_dir: str
    results_dir: str
    groundtruth_dir: str
    grid_dir: str


def _folders(name: str) -> dict:
    return {"wav_dir": names.wav_dir(name), "cochlea_dir": names.cochlea_dir(name),
            "combined_dir": names.combined_dir(name), "results_dir": names.results_dir(name),
            "groundtruth_dir": names.groundtruth_dir(name), "grid_dir": names.grid_dir(name)}


def get_design(name: str = DEFAULT_DESIGN_NAME) -> SweepDesign:
    """The design called ``name`` (one of DESIGN_NAMES); raises ValueError for any other name."""
    names.check_design_name(name)
    if name == ERB_DESIGN_NAME:
        import stimuli.sweep_design_erb as erb
        run_orders = {"build_run_design": erb.build_run_design, "total_run_duration_s": erb.total_run_duration_s,
                      "SILENCE_SEQ_ID": erb.SILENCE_SEQ_ID, "TR_S": erb.TR_S}
    elif name in names.TAU_DESIGN_NAMES:
        from stimuli.sweep_design_harmonic_complex import HarmonicTauRunDesign
        harmonic = HarmonicTauRunDesign(name[len(names.TAU_DESIGN_PREFIX):])
        run_orders = {"build_run_design": harmonic.build_run_design,
                      "total_run_duration_s": harmonic.total_run_duration_s,
                      "SILENCE_SEQ_ID": harmonic.SILENCE_SEQ_ID, "TR_S": harmonic.TR_S}
    else:
        from stimuli.sweep_design_harmonic_complex import HarmonicSweepRunDesign
        harmonic = HarmonicSweepRunDesign(name[-1])
        run_orders = {"build_run_design": harmonic.build_run_design,
                      "total_run_duration_s": harmonic.total_run_duration_s,
                      "SILENCE_SEQ_ID": harmonic.SILENCE_SEQ_ID, "TR_S": harmonic.TR_S}
    return SweepDesign(name=name, label=names.design_label(name), **run_orders, **_folders(name))
