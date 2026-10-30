"""Recombine CF-batch outputs of examples/cochlea_sweep_wavs.py into one file per WAV.

Batch files are named ``<experiment>_cfb<idx>_<soundfileid>.npz`` (one run folder per
task, so they are searched recursively). For every (experiment, soundfileid) group the
PSTHs are concatenated along the CF axis and the population rate is recomputed from
the combined PSTH (a batch's own population_rate_psth covers only its CFs).

The combined file has the same keys as an unbatched run:
soundfileid, cf_list, duration, fiber_types, metadata, time_axis,
population_rate_psth (n_cf, n_bins), population_rate_mean, psth {fiber_type: (n_cf, n_bins)}.

Usage (from the project root):
    python examples/combine_cf_batches.py --run-dir models_output/<wav_dir_name> \
        --expected-n-batches 6
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from auditory_prf.utils.calculate_population_rate import calculate_population_rate

BATCH_FILE_PATTERN = re.compile(r"^(?P<experiment>.+?)_cfb(?P<batch>\d{2})_(?P<soundfileid>.+)\.npz$")


def find_batch_files(run_dir: Path) -> dict:
    """Group batch files by (experiment, soundfileid) -> {batch_idx: path}."""
    groups = defaultdict(dict)
    for path in sorted(run_dir.rglob("*.npz")):
        match = BATCH_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        key = (match.group("experiment"), match.group("soundfileid"))
        batch_idx = int(match.group("batch"))
        assert batch_idx not in groups[key], f"duplicate batch {batch_idx} for {key}: {path}"
        groups[key][batch_idx] = path
    return groups


def load_batch(path: Path) -> dict:
    """Load one batch npz into plain Python objects."""
    with np.load(path, allow_pickle=True) as data:
        return {
            "soundfileid": str(data["soundfileid"]),
            "cf_list": data["cf_list"],
            "duration": float(data["duration"]),
            "fiber_types": data["fiber_types"],
            "metadata": data["metadata"].item(),
            "time_axis": data["time_axis"],
            "psth": data["psth"].item(),
        }


def combine_batches(batch_paths: dict, expected_n_batches: int) -> dict:
    """Concatenate the CF batches of one WAV.

    Parameters
    ----------
    batch_paths : dict
        {batch_idx: Path}; must be exactly 0 ... expected_n_batches - 1.
    expected_n_batches : int
        Number of CF batches the job array was supposed to produce.

    Returns
    -------
    dict
        Same keys as an unbatched result; psth arrays are (n_cf_total, n_bins).
    """
    assert sorted(batch_paths) == list(range(expected_n_batches)), (
        f"batches present {sorted(batch_paths)}, expected 0..{expected_n_batches - 1}")
    batches = [load_batch(batch_paths[idx]) for idx in range(expected_n_batches)]
    first = batches[0]

    for batch in batches[1:]:
        assert batch["soundfileid"] == first["soundfileid"]
        assert batch["duration"] == first["duration"]
        assert np.array_equal(batch["time_axis"], first["time_axis"]), "time axes differ"
        assert list(batch["fiber_types"]) == list(first["fiber_types"])
        assert batch["metadata"] == first["metadata"], "metadata differs between batches"

    cf_list = np.concatenate([batch["cf_list"] for batch in batches])
    assert np.all(np.diff(cf_list) > 0), "CFs not strictly increasing after concatenation"

    psth = {fiber_type: np.concatenate([batch["psth"][fiber_type] for batch in batches], axis=0)
            for fiber_type in first["fiber_types"]}
    for fiber_type, array in psth.items():
        assert array.shape == (len(cf_list), len(first["time_axis"])), (fiber_type, array.shape)

    return {
        "soundfileid": first["soundfileid"],
        "cf_list": cf_list,
        "duration": first["duration"],
        "fiber_types": first["fiber_types"],
        "metadata": first["metadata"],
        "time_axis": first["time_axis"],
        "population_rate_psth": calculate_population_rate(psth),
        "population_rate_mean": None,
        "psth": psth,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True,
                        help="models_output/<wav_dir_name>, searched recursively.")
    parser.add_argument("--expected-n-batches", type=int, required=True,
                        help="CF batches per WAV; fails if any is missing.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Default: <run-dir>/combined.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = find_batch_files(run_dir)
    assert groups, f"no *_cfb<idx>_*.npz files under {run_dir}"
    for (experiment, soundfileid), batch_paths in sorted(groups.items()):
        combined = combine_batches(batch_paths, args.expected_n_batches)
        out_path = out_dir / f"{experiment}_{soundfileid}.npz"
        np.savez_compressed(out_path, **combined)
        print(f"{out_path.name}: {len(combined['cf_list'])} CFs from {len(batch_paths)} batches")


if __name__ == '__main__':
    main()
