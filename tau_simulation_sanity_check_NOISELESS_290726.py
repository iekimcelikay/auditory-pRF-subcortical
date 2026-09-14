import numpy as np
from pathlib import Path
from typing import Optional

from sympy import im

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import build_per_seq_trains
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, convolve_hrf, SUBCORTICAL_PARAMS

# --- Paths ---------------------------------------------======================
RESULTS_DIR = Path("models_output/toneclouds_gaussianprf_20260619_0114")
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"
# ─── Fixed parameters for the simulation –––––––––––––––––––––––––––––––––––––
CF_IDX = 10 # ~448 Hz SINGLE CF FOR NOW
ALPHA = 4.0
# RHO = 1.0 # This parameter is the ratio between ON/OFF channels but unused for the current simulation purposes
TR_S = 1.6 
TOTAL_RUN_DUR_S = 652.8
SIGNAL_DT_S = 1e-3  
RECTIFY = True
N_RUNS = 1
SEQ_DUR_S = 20.0

# ––– Sweep parameters ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
W_SWEEP = [0.8] # Final goal is to sweep over W values but for now keeep things simple 
TAU_GT_SWEEP = [20, 30, 50, 75, 100, 150, 250] # FOR NOW LET'S JUST SIMULATE A SAMPLE OF GROUND TRUTH SIGNALS
# Regression will be done at a later script 

# ––– Phase 1: load cochlear PSTHs -> per_seq 
print(f"\nPhase 1: loading cochlear PSTHs from {RESULTS_DIR} (CF index {CF_IDX})...")
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {RESULTS_DIR}/wav*/")

cf_list_arr: Optional[np.ndarray] = None
