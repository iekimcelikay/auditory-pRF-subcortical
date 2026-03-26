"""Minimal test to isolate where zilany2014 hangs."""
import sys
import numpy as np
print("1. Imports OK", flush=True)

from cochlea.zilany2014 import _zilany2014
print("2. _zilany2014 C extension imported OK", flush=True)

# Generate a simple 50ms tone at 1kHz
fs = 100000
dur = 0.05
t = np.arange(0, dur, 1/fs)
tone = 0.01 * np.sin(2 * np.pi * 1000 * t)  # ~60 dB SPL
print(f"3. Tone generated: {tone.shape}, fs={fs}", flush=True)

# Step 1: run_ihc (BM + IHC model)
print("4. Calling _zilany2014.run_ihc()...", flush=True)
vihc = _zilany2014.run_ihc(
    signal=tone,
    cf=1000.0,
    fs=fs,
    species='human',
    cohc=1.0,
    cihc=1.0
)
print(f"5. run_ihc DONE: vihc shape={vihc.shape}", flush=True)

# Step 2: run_synapse (1 fiber, no ffGn)
print("6. Calling _zilany2014.run_synapse(ffGn=False)...", flush=True)
synout = _zilany2014.run_synapse(
    fs=fs,
    vihc=vihc,
    cf=1000.0,
    anf_type='hsr',
    powerlaw='approximate',
    ffGn=False
)
print(f"7. run_synapse DONE: synout shape={synout.shape}", flush=True)

# Step 3: run_spike_generator
print("8. Calling _zilany2014.run_spike_generator()...", flush=True)
spikes = _zilany2014.run_spike_generator(
    synout=synout,
    fs=fs,
)
print(f"9. run_spike_generator DONE: {len(spikes)} spikes", flush=True)

print("10. ALL STEPS COMPLETED SUCCESSFULLY", flush=True)
