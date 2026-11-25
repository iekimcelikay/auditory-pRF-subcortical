"""Put the label of a sweep design (e.g. 'Harmonic sweeps: 3 rates x 2 tones x 4 ISIs') into the titles of plots.

The labels come from stimuli/sweep_design_names.py; the recovery scripts pass them to their plot functions through an
optional ``design_label`` argument (None = the plot looks exactly as before).
"""

from typing import Optional


def titled(title: str, design_label: Optional[str]) -> str:
    """``title`` with the design label on a second line; ``title`` unchanged when there is no label."""
    return title if not design_label else f"{title}\n{design_label}"


def label_only_suptitle(fig, design_label: Optional[str], fontsize: int = 10) -> None:
    """For figures that have no title of their own: the design label as the figure title (nothing if None)."""
    if design_label:
        fig.suptitle(design_label, fontsize=fontsize, y=1.02)
