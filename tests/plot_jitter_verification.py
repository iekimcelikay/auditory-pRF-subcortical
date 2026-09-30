"""
plot_jitter_verification.py
Compares MATLAB vs Python jitter-perturbed (frequency, amplitude) recovery
across 100 seeded draws per component/jitter level, since sample-by-sample
comparison is impossible for jittered (random) components -- see
tests/jitter_recovery_utils.py's module docstring.

Run from project root:
    python tests/plot_jitter_verification.py

Requires tests/matlab_verify_jitter.m to have been run first (MATLAB CSVs
present in tests/matlab_verification/).

Figures saved to tests/matlab_verification/:
    fig4_jitter_j0.10.png
    fig4_jitter_j0.50.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from jitter_recovery_utils import (
    NOMINAL, JITTER_LEVELS, N_SEEDS, drift_n_basis_valid, cohens_d,
    recover_cardiac_or_respiratory, recover_drift,
)

MATLAB_DIR = Path(__file__).parent / 'matlab_verification'
COMPONENTS = ['cardiac', 'respiratory', 'drift']
PARAMS = ['f', 'a']
PARAM_LABELS = {'f': 'frequency (Hz)', 'a': 'amplitude'}


def python_recovered(component: str, jitter: float) -> pd.DataFrame:
    """Recover (f, a, valid) for all N_SEEDS draws of one component/jitter level."""
    nom = NOMINAL[component]
    rows = []
    for seed in range(N_SEEDS):
        if component == 'drift':
            f, a = recover_drift(seed, jitter, nom['freq'], nom['amp'])
            valid = drift_n_basis_valid(f)
        else:
            f, a, _ = recover_cardiac_or_respiratory(seed, jitter, nom['freq'], nom['amp'])
            valid = True
        rows.append((seed, f, a, valid))
    return pd.DataFrame(rows, columns=['seed', 'f', 'a', 'valid'])


print("── Jitter recovery: MATLAB vs Python ──────────────────────────────────")

summary_rows = []

for jitter in JITTER_LEVELS:
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    fig.suptitle(
        f'Jitter recovery — MATLAB vs Python (jitter = {jitter})\n'
        f'{N_SEEDS} seeds/language  |  mean ± SD, superimposed violins, Cohen\'s d',
        fontsize=12,
    )

    for row, component in enumerate(COMPONENTS):
        nom = NOMINAL[component]
        ml_path = MATLAB_DIR / f'jitter_{component}_j{jitter:.2f}.csv'
        df_ml = pd.read_csv(ml_path)
        df_ml = df_ml[df_ml['valid'].astype(bool)]
        df_py = python_recovered(component, jitter)
        df_py = df_py[df_py['valid']]

        n_ml, n_py = len(df_ml), len(df_py)
        print(f"\n{component} (jitter={jitter}):  n_valid  MATLAB={n_ml}/{N_SEEDS}  Python={n_py}/{N_SEEDS}")

        for col, param in enumerate(PARAMS):
            ax = axes[row, col]
            ml_vals = df_ml[param].to_numpy()
            py_vals = df_py[param].to_numpy()

            target_mean = nom[{'f': 'freq', 'a': 'amp'}[param]]
            target_sd = jitter * target_mean

            d = cohens_d(py_vals, ml_vals)

            print(f"  {param:>1s}  target={target_mean:9.4f}±{target_sd:.4f}  "
                  f"MATLAB={ml_vals.mean():9.4f}±{ml_vals.std(ddof=1):.4f}  "
                  f"Python={py_vals.mean():9.4f}±{py_vals.std(ddof=1):.4f}  "
                  f"Cohen's d={d:+.3f}")

            summary_rows.append(dict(
                jitter=jitter, component=component, param=param,
                target_mean=target_mean, target_sd=target_sd,
                matlab_n=n_ml, matlab_mean=ml_vals.mean(), matlab_sd=ml_vals.std(ddof=1),
                python_n=n_py, python_mean=py_vals.mean(), python_sd=py_vals.std(ddof=1),
                cohens_d=d,
            ))

            # Violins
            parts = ax.violinplot([ml_vals, py_vals], positions=[0, 1], widths=0.7,
                                   showmeans=False, showextrema=False)
            for pc, color in zip(parts['bodies'], ['firebrick', 'steelblue']):
                pc.set_facecolor(color)
                pc.set_alpha(0.35)

            # Mean ± SD markers
            for x, vals, color in [(0, ml_vals, 'firebrick'), (1, py_vals, 'steelblue')]:
                ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt='o', color=color,
                            capsize=6, ms=8, zorder=5)

            # Theoretical target
            ax.axhline(target_mean, color='k', ls='--', lw=1.2, alpha=0.7, label='target mean')
            ax.axhspan(target_mean - target_sd, target_mean + target_sd, color='k', alpha=0.08)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'MATLAB\n(n={n_ml})', f'Python\n(n={n_py})'])
            ax.set_ylabel(PARAM_LABELS[param])
            ax.set_title(f'{component} — {param}   (Cohen\'s d = {d:+.3f})', fontsize=10)
            ax.grid(True, alpha=0.3)

    legend_handles = [
        Patch(facecolor='firebrick', alpha=0.35, label='MATLAB draws (violin)'),
        Patch(facecolor='steelblue', alpha=0.35, label='Python draws (violin)'),
        Line2D([0], [0], marker='o', color='firebrick', ms=8, lw=1.5,
               label='MATLAB mean ± SD'),
        Line2D([0], [0], marker='o', color='steelblue', ms=8, lw=1.5,
               label='Python mean ± SD'),
        Line2D([0], [0], color='k', ls='--', lw=1.2, alpha=0.7,
               label='theoretical target mean'),
        Patch(facecolor='k', alpha=0.08, label='theoretical target ± 1 SD'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.015), fontsize=9, frameon=True)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = MATLAB_DIR / f'fig4_jitter_j{jitter:.2f}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"\nSaved {out}")
    plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
csv_out = MATLAB_DIR / 'jitter_verification_summary.csv'
summary_df.to_csv(csv_out, index=False, float_format='%.6f')
print(f"\nSaved {csv_out}")

txt_out = MATLAB_DIR / 'jitter_verification_summary.txt'
with open(txt_out, 'w') as fh:
    fh.write("Jitter recovery: MATLAB vs Python -- summary\n")
    fh.write("=" * 78 + "\n")
    fh.write(
        "Recovered (frequency, amplitude) from 100 seeded jitter draws per\n"
        "component/jitter level, comparing MATLAB's gari_pmNoise.m against the\n"
        "Python port. Sample-by-sample comparison is not possible (MATLAB's\n"
        "randn and NumPy's standard_normal diverge from an identical seed --\n"
        "see tests/jitter_recovery_utils.py); populations are compared instead\n"
        "via mean +/- SD against the theoretical target (nominal x jitter), plus\n"
        "Cohen's d. |d| < 0.2 is conventionally a negligible/small effect.\n"
    )
    for jitter in JITTER_LEVELS:
        fh.write("\n" + "-" * 78 + f"\njitter = {jitter}\n" + "-" * 78 + "\n")
        sub = summary_df[summary_df['jitter'] == jitter]
        for _, r in sub.iterrows():
            fh.write(
                f"  {r['component']:11s} {r['param']}   "
                f"target={r['target_mean']:10.4f} +/- {r['target_sd']:.4f}   "
                f"MATLAB (n={int(r['matlab_n']):3d})={r['matlab_mean']:10.4f} +/- {r['matlab_sd']:.4f}   "
                f"Python (n={int(r['python_n']):3d})={r['python_mean']:10.4f} +/- {r['python_sd']:.4f}   "
                f"Cohen's d={r['cohens_d']:+.3f}\n"
            )
print(f"Saved {txt_out}")

print(f"\nAll outputs in: {MATLAB_DIR}")
