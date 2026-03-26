"""Custom colormaps for auditory pRF visualizations."""

from matplotlib.colors import LinearSegmentedColormap

# Auditory cortex — teal-to-red diverging scheme (8 anchor colors)
_AUDITORY_CORTEX_COLORS = [
    (0.000, (0 / 255, 95 / 255, 115 / 255)),
    (0.143, (10 / 255, 147 / 255, 150 / 255)),
    (0.286, (148 / 255, 210 / 255, 189 / 255)),
    (0.429, (233 / 255, 216 / 255, 166 / 255)),
    (0.571, (238 / 255, 155 / 255, 0 / 255)),
    (0.714, (202 / 255, 103 / 255, 2 / 255)),
    (0.857, (187 / 255, 62 / 255, 3 / 255)),
    (1.000, (174 / 255, 32 / 255, 18 / 255)),
]

auditory_cortex_cmap = LinearSegmentedColormap.from_list(
    "auditory_cortex", _AUDITORY_CORTEX_COLORS
)
