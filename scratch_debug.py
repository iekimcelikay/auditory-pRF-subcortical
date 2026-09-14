from pathlib import Path

import numpy as np
import torch 

# 18/08/2026
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.utils.result_saver import ResultSaver
from torch_functions import soft_select_channel_linear, apply_powerlaw_torch

# functions
def load_cf_list(npz_path: Path) -> np.ndarray:
	"""Load the (n_cfs,) array of channel CFs in Hz from one .npz file."""
	data = ResultSaver(npz_path.parent).load_npz(npz_path.name)
	return np.asarray(data["cf_list"])


# ── experiment defaults ───────────────────────────────────────────────────────
EXP_NAME         = "toneclouds_gaussianprf_20260619_0114"
DEFAULT_BASE_DIR  = Path(f"./models_output/{EXP_NAME}")
cf = 10 
_results_dir = DEFAULT_BASE_DIR
_results_dir = Path(_results_dir).expanduser().resolve()

DATE_MATCH = "20260626"

npz_files = sorted(
    p for p in _results_dir.glob("wav*/**/*.npz")
    if any(DATE_MATCH in part for part in p.parts)
)

if not npz_files:
	raise FileNotFoundError(
		f"No .npz files found in {_results_dir}/wav*/. "
		"Run the cochlear simulation first and check results_dir.")
print(len(npz_files))


# cf_list is the same cochlear channel grid for every stimulus file, so pull
# it once (not inside the per-file loop below).
cf_list = load_cf_list(npz_files[0])
print(cf_list.shape, cf_list[:5])

# CONVERT TO PYTORCH TENSOR 
cf_list = torch.from_numpy(cf_list).float()
print(type(cf_list))

# 2. Load population psth .npz files

for i, npz_path in enumerate(npz_files, 1):
	population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, cf)
	print(seq_id)
	print(population_psth.shape)
	# CONVERT TO PYTORCH TENSOR

# load_population_psth works fine.





