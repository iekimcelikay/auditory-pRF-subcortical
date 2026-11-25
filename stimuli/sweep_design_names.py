"""Names, labels and folder rules of the FM-sweep designs: the one place every script reads them from.

One rule: everything of a design lives under its name.

    WAVs                       stimuli/produced/<name>/
    cochlear results           models_output/<name>/   (the combined PSTHs in models_output/<name>/combined/)
    recovery results           sweep_recovery_results/<name>/<timestamp>_...
    ground-truth results       sweep_groundtruth_results/<name>/<timestamp>_...
    candidate grids            sweep_recovery_multirun_grids/<name>/

Designs
    sweep_design_erb3band    3 ERB bands x 8 rates, continuous sweeps (stimuli/sweep_design_erb.py); the default
    harmonic_sweep_armA      harmonic sweeps with silences, 3 rates x 2 tones x 4 ISIs   (stimuli/sweep_design_harmonic_complex.py)
    harmonic_sweep_armB      harmonic sweeps with silences, 3 rates x 4 tones x 2 ISIs

The label of a design (``design_label``) is what plots and result metadata show, so that "armA" is never the only
thing a reader sees. This module has no heavy imports (no numpy, no pipeline), so the cochlea runner, the renderer and
the plotting scripts can use it anywhere.
"""

ERB_DESIGN_NAME = "sweep_design_erb3band"
HARMONIC_DESIGN_PREFIX = "harmonic_sweep_arm"
HARMONIC_LAYOUT_COUNTS = {"A": {"n_tones": 2, "n_isis": 4}, "B": {"n_tones": 4, "n_isis": 2}}
N_RATES = 3

DESIGN_NAMES = (ERB_DESIGN_NAME,) + tuple(f"{HARMONIC_DESIGN_PREFIX}{layout}" for layout in HARMONIC_LAYOUT_COUNTS)
DEFAULT_DESIGN_NAME = ERB_DESIGN_NAME


def harmonic_design_name(layout: str) -> str:
    """'A' -> 'harmonic_sweep_armA'."""
    assert layout in HARMONIC_LAYOUT_COUNTS, layout
    return f"{HARMONIC_DESIGN_PREFIX}{layout}"


def check_design_name(name: str) -> str:
    """Return name if it is a known design, else raise ValueError listing the choices."""
    if name not in DESIGN_NAMES:
        raise ValueError(f"unknown design {name!r}; choose one of {DESIGN_NAMES}")
    return name


def design_label(name: str) -> str:
    """Human-readable label of a design, shown in plots and stored in result metadata."""
    check_design_name(name)
    if name == ERB_DESIGN_NAME:
        return "ERB 3 bands x 8 rates (continuous sweeps)"
    counts = HARMONIC_LAYOUT_COUNTS[name[len(HARMONIC_DESIGN_PREFIX):]]
    return f"Harmonic sweeps: {N_RATES} rates x {counts['n_tones']} tones x {counts['n_isis']} ISIs"


# ── Folder rules (project-root-relative strings; callers join them to their root) ─────────────────────────────
def wav_dir(name: str) -> str:
    return f"stimuli/produced/{check_design_name(name)}"


def cochlea_dir(name: str) -> str:
    return f"models_output/{check_design_name(name)}"


def combined_dir(name: str) -> str:
    return f"{cochlea_dir(name)}/combined"


def results_dir(name: str) -> str:
    return f"sweep_recovery_results/{check_design_name(name)}"


def groundtruth_dir(name: str) -> str:
    return f"sweep_groundtruth_results/{check_design_name(name)}"


def grid_dir(name: str) -> str:
    return f"sweep_recovery_multirun_grids/{check_design_name(name)}"
