import numpy as np
import pandas as pd
import time
import os
import gc
import thorns as th
from subcorticalSTRF.stim_generator.soundgen import SoundGen

from cochlea.zilany2014 import calc_cfs

# ---------------- Parameters (reduced for speed) ----------------
PERIPHERAL_FS = 100e3
SAMPLE_RATE   = 100e3
DURATION      = 0.200        # shorter stimulus
TAU_RAMP      = 0.005
NUM_CF        = 5            # fewer CFs
MIN_CF        = 125
MAX_CF        = 2500
FREQS         = [700]       # single test frequency
DBS           = [70]         # single level
NUM_PER_TYPE  = 32           # fibers per type in population condition
NUM_REPS      = NUM_PER_TYPE # number of (1,1,1) repetitions per type to match
BASE_SEED     = 123456789

# ---------------- Helpers ----------------
FIBER_TYPE_ORDER = ['lsr','msr','hsr']

def make_tone(freq, level_db, sr, dur, ramp, n_harm=1, harmonic_factor=0.5):
    t = np.arange(int(dur*sr)) / sr
    tone = np.sin(2*np.pi*freq*t)
    # simple raised-cosine ramp
    ramp_len = int(ramp*sr)
    if ramp_len > 0:
        w = 0.5*(1-np.cos(np.linspace(0, np.pi, ramp_len)))
        tone[:ramp_len] *= w
        tone[-ramp_len:] *= w[::-1]
    # level scaling (arbitrary: treat as linear scaling from dB)
    tone = tone * 10**(level_db/20 - 1)  # relative; consistent across approaches
    return tone.astype(np.float64)

def extract_rates(trains_acc, spike_matrix, duration):
    # trains_acc sorted by cf,type already
    records = []
    for i in range(len(trains_acc)):
        row = trains_acc.iloc[i]
        spikes = spike_matrix[i]
        spike_count = spikes.sum()
        rate = spike_count / duration
        records.append( (row['cf'], row['type'], rate) )
    return records

def aggregate_stats(records):
    # records: list of (cf, type, rate)
    df = pd.DataFrame(records, columns=['cf','type','rate'])
    grouped = df.groupby(['cf','type'])['rate'].agg(['mean','var','count']).reset_index()
    grouped.rename(columns={'mean':'mean_rate','var':'var_rate','count':'n'}, inplace=True)
    return grouped

def run_population(sound_gen, cfs, freqs, dbs, seed):
    np.random.seed(seed)
    all_rec = []
    for db in dbs:
        for freq in freqs:
            tone = make_tone(freq, db, SAMPLE_RATE, DURATION, TAU_RAMP)
            trains_acc = sound_gen.cochlea_rate_manual(
                tone,
                PERIPHERAL_FS,
                MIN_CF,
                MAX_CF,
                NUM_CF,
                (NUM_PER_TYPE, NUM_PER_TYPE, NUM_PER_TYPE),
                seed=seed
            )
            trains_acc.sort_values(['cf','type'], inplace=True)
            spike_mat = th.trains_to_array(trains_acc, PERIPHERAL_FS).T
            all_rec.extend(extract_rates(trains_acc, spike_mat, DURATION))
            del trains_acc, spike_mat, tone
            gc.collect()
    return aggregate_stats(all_rec)

def run_aggregated(sound_gen, cfs, freqs, dbs, base_seed):
    all_rec = []
    for rep in range(NUM_REPS):
        seed = base_seed + 1000 + rep
        np.random.seed(seed)
        for db in dbs:
            for freq in freqs:
                tone = make_tone(freq, db, SAMPLE_RATE, DURATION, TAU_RAMP)
                trains_acc = sound_gen.cochlea_rate_manual(
                    tone,
                    PERIPHERAL_FS,
                    MIN_CF,
                    MAX_CF,
                    NUM_CF,
                    (1,1,1),
                    seed=seed
                )
                trains_acc.sort_values(['cf','type'], inplace=True)
                spike_mat = th.trains_to_array(trains_acc, PERIPHERAL_FS).T
                all_rec.extend(extract_rates(trains_acc, spike_mat, DURATION))
                del trains_acc, spike_mat, tone
        gc.collect()
    return aggregate_stats(all_rec)

def compare(pop_df, agg_df):
    merged = pop_df.merge(
        agg_df,
        on=['cf','type'],
        suffixes=('_pop','_agg'),
        how='inner'
    )
    merged['mean_diff'] = merged['mean_rate_agg'] - merged['mean_rate_pop']
    merged['mean_rel_pct'] = 100 * merged['mean_diff'] / merged['mean_rate_pop'].replace(0, np.nan)
    merged['var_ratio'] = merged['var_rate_agg'] / merged['var_rate_pop'].replace(0, np.nan)
    return merged

# ---------------- Main ----------------
def main():
    cfs = calc_cfs((MIN_CF, MAX_CF, NUM_CF), species='human')
    sound_gen = SoundGen(SAMPLE_RATE, TAU_RAMP)

    t0 = time.time()
    pop_stats = run_population(sound_gen, cfs, FREQS, DBS, BASE_SEED)
    t1 = time.time()
    agg_stats = run_aggregated(sound_gen, cfs, FREQS, DBS, BASE_SEED)
    t2 = time.time()

    comp = compare(pop_stats, agg_stats)
    comp.sort_values(['cf','type'], inplace=True)

    print("\nPopulation run time: {:.2f}s".format(t1 - t0))
    print("Aggregated runs time: {:.2f}s".format(t2 - t1))
    print("\nComparison (first rows):")
    print(comp.head())

    print("\nMax abs mean diff (sp/s): {:.4f}".format(comp['mean_diff'].abs().max()))
    print("Max abs mean rel diff (%): {:.2f}".format(comp['mean_rel_pct'].abs().max()))
    print("Var ratio range: {:.3f} .. {:.3f}".format(comp['var_ratio'].min(), comp['var_ratio'].max()))

    out_csv = "mean_var_comparison.csv"
    comp.to_csv(out_csv, index=False)
    print("\nSaved detailed comparison to {}".format(out_csv))

if __name__ == "__main__":
    main()