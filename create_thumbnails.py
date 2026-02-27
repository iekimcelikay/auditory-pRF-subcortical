from PIL import Image
from pathlib import Path

src = Path("/home/ekim/auditory-pRF-subcortical/figures/dipc_test_250225_01")
dst = src.parent / "dipc_test_250225_01_thumbs"
dst.mkdir(exist_ok=True)

for p in src.glob("*.png"):
    img = Image.open(p)
    img.thumbnail((800, 600))  # adjust to taste
    img.save(dst / p.name, optimize=True)