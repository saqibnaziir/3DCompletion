"""
Aggregate + paired statistics for measurement 1, recomputed from the per-mesh CSV
that rev_depth_deciles.py already wrote. Pure post-processing: no sampling, no
GPU, no checkpoint access -- so it is safe to run while other jobs hold the GPU.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import csv
import json
import argparse
import numpy as np
from scipy import stats

ORDER = ['eps', 'x0', 'x0+depth', 'x0+depth+nrm']


def paired(a, b, label_a, label_b):
    d = a - b
    t, p_t = stats.ttest_rel(a, b)
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p_w = float('nan')
    sd = d.std(ddof=1)
    return {
        'arm': label_a, 'ref': label_b,
        'arm_mean_mm': float(a.mean()), 'ref_mean_mm': float(b.mean()),
        'mean_diff_mm': float(d.mean()),
        'pct_change': float(100.0 * d.mean() / b.mean()),
        'n_meshes': int(len(a)), 'wins_for_arm': int((d < 0).sum()),
        't': float(t), 'p_ttest': float(p_t), 'p_wilcoxon': p_w,
        'cohens_dz': float(d.mean() / sd) if sd > 0 else float('nan'),
        'ci95_lo_mm': float(d.mean() - 1.96 * sd / np.sqrt(len(d))),
        'ci95_hi_mm': float(d.mean() + 1.96 * sd / np.sqrt(len(d))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv',
                    default=os.path.join('data', '_revision',
                                         'depth_deciles_per_mesh.csv'))
    ap.add_argument('--out_dir', default=os.path.join('data', '_revision'))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.in_csv, newline='')))
    for r in rows:
        for k, v in r.items():
            if k not in ('arm', 'sample'):
                r[k] = float(v) if v not in ('', 'nan') else float('nan')

    labels = [l for l in ORDER if any(r['arm'] == l for r in rows)]
    per = {l: {r['sample']: r for r in rows if r['arm'] == l} for l in labels}
    samples = sorted(per[labels[0]])
    degen = sorted({r['sample'] for r in rows if r['degenerate_depth'] == 1.0})
    ok = [s for s in samples if s not in degen]

    def vec(l, key, subset):
        return np.array([per[l][s][key] for s in subset], dtype=np.float64)

    W = 100
    print("=" * W)
    print("TABLE 1  HOLE ERROR BY GEODESIC-DEPTH DECILE (mm; K=10, seed 0, "
          "50 DDIM steps, EMA weights)")
    print(f"  'all missing' over all {len(samples)} test meshes; deciles over the "
          f"{len(ok)} with a defined depth field")
    print(f"  depth-undefined (hole = entire disconnected component): "
          f"{', '.join(degen)}")
    print("-" * W)
    print(f"{'arm':<15}{'all missing':>14}{'all (n=%d)' % len(ok):>13}"
          f"{'deepest 10%':>14}{'shallowest 10%':>17}{'deep/shallow':>15}")
    summary = {}
    for l in labels:
        a20 = vec(l, 'err_all_mm', samples)
        a17 = vec(l, 'err_all_mm', ok)
        d = vec(l, 'err_deep_mm', ok)
        s = vec(l, 'err_shallow_mm', ok)
        summary[l] = {
            'all_mm_all_meshes': float(a20.mean()),
            'all_sd_all_meshes': float(a20.std(ddof=1)),
            'all_mm': float(a17.mean()), 'all_sd': float(a17.std(ddof=1)),
            'deep_mm': float(d.mean()), 'deep_sd': float(d.std(ddof=1)),
            'shallow_mm': float(s.mean()), 'shallow_sd': float(s.std(ddof=1)),
            'ratio_deep_over_shallow': float(d.mean() / s.mean()),
            'n_all': len(samples), 'n_decile': len(ok)}
        print(f"{l:<15}{a20.mean():>14.3f}{a17.mean():>13.3f}{d.mean():>14.3f}"
              f"{s.mean():>17.3f}{d.mean()/s.mean():>14.2f}x")
    print("=" * W)

    # ---- the requested test, plus the same contrast on the other two columns ----
    tests = []
    print(f"\nTABLE 2  PAIRED TESTS, x0+depth vs x0 (n={len(ok)} meshes, "
          f"paired per mesh)")
    print("-" * W)
    print(f"{'region':<18}{'x0':>10}{'x0+depth':>11}{'diff':>10}{'%':>9}"
          f"{'wins':>7}{'t':>8}{'p(t)':>10}{'p(wilc)':>10}{'dz':>8}")
    for key, name in [('err_deep_mm', 'deepest decile'),
                      ('err_all_mm', 'all missing'),
                      ('err_shallow_mm', 'shallowest decile')]:
        r = paired(vec('x0+depth', key, ok), vec('x0', key, ok), 'x0+depth', 'x0')
        r['region'] = name
        tests.append(r)
        print(f"{name:<18}{r['ref_mean_mm']:>10.3f}{r['arm_mean_mm']:>11.3f}"
              f"{r['mean_diff_mm']:>+10.3f}{r['pct_change']:>+8.1f}%"
              f"{r['wins_for_arm']:>4}/{len(ok):<2}{r['t']:>8.3f}"
              f"{r['p_ttest']:>10.4f}{r['p_wilcoxon']:>10.4f}{r['cohens_dz']:>+8.3f}")
    hd = tests[0]
    print(f"\n  deepest-decile 95% CI on the difference: "
          f"[{hd['ci95_lo_mm']:+.3f}, {hd['ci95_hi_mm']:+.3f}] mm")

    # supporting contrast: the normal-loss arm on the deepest decile
    extra = []
    if 'x0+depth+nrm' in per:
        for ref in ('x0', 'x0+depth'):
            r = paired(vec('x0+depth+nrm', 'err_deep_mm', ok),
                       vec(ref, 'err_deep_mm', ok), 'x0+depth+nrm', ref)
            r['region'] = 'deepest decile'
            extra.append(r)
        print(f"\nTABLE 3  PAIRED TESTS on the deepest decile, x0+depth+nrm vs others")
        print("-" * W)
        print(f"{'contrast':<26}{'diff':>10}{'%':>9}{'wins':>8}{'t':>9}"
              f"{'p(t)':>10}{'p(wilc)':>10}")
        for r in extra:
            print(f"{r['arm']+' vs '+r['ref']:<26}{r['mean_diff_mm']:>+10.3f}"
                  f"{r['pct_change']:>+8.1f}%{r['wins_for_arm']:>5}/{len(ok):<2}"
                  f"{r['t']:>9.3f}{r['p_ttest']:>10.4f}{r['p_wilcoxon']:>10.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'depth_deciles_summary.json'), 'w') as f:
        json.dump({'summary': summary, 'tests_x0depth_vs_x0': tests,
                   'tests_nrm_deepest': extra,
                   'degenerate_meshes': degen,
                   'n_meshes_all': len(samples), 'n_meshes_decile': len(ok),
                   'K': 10, 'steps': 50, 'seed': 0, 'decile_frac': 0.1}, f, indent=1)
    print(f"\nwrote {args.out_dir}/depth_deciles_summary.json")


if __name__ == '__main__':
    main()
