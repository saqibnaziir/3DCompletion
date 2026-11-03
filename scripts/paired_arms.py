"""
Paired statistics across diffusion arms, from diag_sampling_floor CSVs.

Rebuild of the 08-01 scratchpad/paired_ctrl_nrm.py (that file lived in a
scratchpad that was wiped), generalized to an arbitrary set of arms.

Every arm must have been scored with scripts/diag_sampling_floor.py at the SAME
--K and on the SAME test meshes; rows are matched by the `sample` column, so the
comparison is paired per mesh. Arms are compared with a paired t-test and a
Wilcoxon signed-rank test (n=20 is small; if the two disagree, trust Wilcoxon).

WHICH COLUMN TO REPORT
  meanK_*   error of the mean-of-K mesh   <- the numbers in the handoff tables
  single_*  E_k[ error(sample_k) ]        <- one draw, averaged over draws
These are DIFFERENT quantities, not two estimates of one thing. single_ is what
the model actually outputs; meanK_ is the Monte-Carlo conditional mean and is
NOT a valid diffusion sample. Report both and say which is which -- the 08-01
handoff table calls the meanK_ numbers "expected over 10 draws", which is
misleading phrasing for what they measure.

Never compare arms on a single saved mesh (see the 07-31b lesson: single-draw
noise flipped the sign of the result).

Run (conda 3dmesh python):
  python scripts/paired_arms.py \
      --arm "parent:data/completion_test_samples/test_set/diag_sampling_floor" \
      --arm "ctrl:data/completion_test_samples/test_set/diag_sampling_floor_ctrl" \
      --arm "nrm316:data/completion_test_samples/test_set/diag_sampling_floor_nrm316" \
      --ref ctrl
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import numpy as np
from scipy import stats

# (csv column, label, unit, lower_is_better)
METRICS = [
    ('meanK_nrm_deg',   'normals, hole  (mean-of-K)', 'deg', True),
    ('meanK_seam_deg',  'normals, seam  (mean-of-K)', 'deg', True),
    ('meanK_pos_mm',    'position, hole (mean-of-K)', 'mm',  True),
    ('single_nrm_deg',  'normals, hole  (single)',    'deg', True),
    ('single_seam_deg', 'normals, seam  (single)',    'deg', True),
    ('single_pos_mm',   'position, hole (single)',    'mm',  True),
    ('diversity_mm',    'sampling spread',            'mm',  None),
    ('meanK_rough',     'roughness ratio (mean-of-K)', '',   None),
]


def load_arm(path):
    """path may be the diag folder or the csv itself -> {sample: {col: float}}."""
    if os.path.isdir(path):
        path = os.path.join(path, 'diag_sampling_floor.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing csv: {path}")
    out = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            out[row['sample']] = {k: v for k, v in row.items() if k != 'sample'}
    if not out:
        raise SystemExit(f"empty csv: {path}")
    return out


def col(arm, samples, name):
    return np.array([float(arm[s][name]) for s in samples], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, metavar='NAME:PATH',
                    help='repeatable; PATH is a diag_sampling_floor folder or csv')
    ap.add_argument('--ref', default=None,
                    help='arm every other arm is contrasted against '
                         '(default: the first --arm)')
    ap.add_argument('--per_mesh', default=None,
                    help='metric column to print a per-mesh table for '
                         '(e.g. meanK_nrm_deg)')
    ap.add_argument('--out_csv', default=None, help='write the contrast table here')
    args = ap.parse_args()

    arms = {}
    order = []
    for spec in args.arm:
        if ':' not in spec:
            raise SystemExit(f"--arm needs NAME:PATH, got {spec!r}")
        name, path = spec.split(':', 1)
        arms[name] = load_arm(path)
        order.append(name)

    ref = args.ref or order[0]
    if ref not in arms:
        raise SystemExit(f"--ref {ref!r} is not one of {order}")

    # pair on the meshes every arm actually scored
    samples = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not samples:
        raise SystemExit("no test meshes are common to all arms")
    for name in order:
        extra = len(arms[name]) - len(samples)
        if extra:
            print(f"  note: {name} has {extra} mesh(es) not shared by all arms; dropped")

    ks = {name: sorted({int(float(r['K'])) for r in arms[name].values()})
          for name in order}
    if len({tuple(v) for v in ks.values()}) != 1:
        print(f"  WARNING: arms were scored at different K: {ks} -- "
              f"spread and mean-of-K are not comparable across different K.")

    n = len(samples)
    print(f"\npaired over {n} meshes; K={ks[ref]}; reference arm = {ref}\n")

    # ---- arm means -------------------------------------------------------
    w = max(len(m[1]) for m in METRICS)
    print("ARM MEANS")
    print(f"  {'metric':{w}}  " + "  ".join(f"{name:>10}" for name in order))
    for c, label, unit, _ in METRICS:
        vals = [f"{col(arms[name], samples, c).mean():10.3f}" for name in order]
        print(f"  {label:{w}}  " + "  ".join(vals) + (f"  {unit}" if unit else ""))

    # ---- paired contrasts ------------------------------------------------
    rows = []
    for name in order:
        if name == ref:
            continue
        print(f"\nPAIRED CONTRAST: {name} vs {ref}")
        print(f"  {'metric':{w}}  {'change':>9}  {'better':>7}  "
              f"{'t':>7}  {'p(t)':>9}  {'p(wilcox)':>9}")
        for c, label, unit, lower_better in METRICS:
            a = col(arms[name], samples, c)
            b = col(arms[ref], samples, c)
            d = a - b
            pct = 100.0 * d.mean() / b.mean() if b.mean() != 0 else float('nan')

            if np.allclose(d, 0):
                t = p_t = p_w = float('nan')
                wins = 0
            else:
                t, p_t = stats.ttest_rel(a, b)
                try:
                    p_w = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p_w = float('nan')
                wins = int((d < 0).sum() if lower_better else (d > 0).sum())

            flag = '' if lower_better is None else ('  *' if p_t < 0.05 else '')
            better = f"{wins}/{n}" if lower_better is not None else '-'
            print(f"  {label:{w}}  {pct:+8.1f}%  {better:>7}  "
                  f"{t:7.2f}  {p_t:9.4f}  {p_w:9.4f}{flag}")
            rows.append(dict(arm=name, ref=ref, metric=c, label=label,
                             arm_mean=a.mean(), ref_mean=b.mean(),
                             change_pct=pct, better=better, t=t,
                             p_ttest=p_t, p_wilcoxon=p_w, n=n))
        print("  (* = p<0.05 on the paired t-test; 'better' counts meshes where "
              "the arm beats the reference)")

    # ---- optional per-mesh table ----------------------------------------
    if args.per_mesh:
        c = args.per_mesh
        print(f"\nPER-MESH: {c}")
        print(f"  {'mesh':10}  " + "  ".join(f"{name:>10}" for name in order))
        for s in samples:
            vals = "  ".join(f"{float(arms[name][s][c]):10.3f}" for name in order)
            print(f"  {s:10}  {vals}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, 'w', newline='') as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.out_csv}")


if __name__ == '__main__':
    main()
