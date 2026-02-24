# Installation

## Requirements
- Python 3.9
- conda environment: `subcorticalSTRF3.9` (see `environment.yaml`)

## First-time setup

```bash
# 1. Build and install the vendored cochlea Cython library
pip install -e ./cochlea/

# 2. Install this package in editable mode
pip install -e .
```

After this, `auditory_prf` is importable from anywhere — no path setup needed.

## On a new machine (e.g. DIPC)

```bash
conda env create -f environment.yaml
conda activate subcorticalSTRF3.9
pip install -e ./cochlea/
pip install -e .
```

---

## When do I need to re-run `pip install -e .`?

| Change | Re-install needed? |
|---|---|
| Edit any `.py` file | **No** — editable install reflects changes immediately |
| Add a new `.py` file to an existing folder | **No** |
| Add a **new subfolder/subpackage** inside `auditory_prf/` | **Yes** |
| Edit `pyproject.toml` | **Yes** |
| Move the project to a different directory | **Yes** |
