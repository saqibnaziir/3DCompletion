"""
Quantitative comparison of completion arms (e.g. eps vs x0) against GT.

TWO FAMILIES OF METRIC, and they measure genuinely different failures:

1. POSITIONAL -- per-vertex Euclidean (L2) distance to GT, exploiting dense
   correspondence (same topology, vertex i <-> vertex i). Reported over three
   regions: missing (masked hole), overall, known. Same metric as
   scripts/evaluate_completion.py:compute_metrics.

2. ORIENTATION (added 07-31, Olivier's suggestion) -- angle between predicted
   and GT vertex normals, in degrees, plus normal consistency (mean cosine).
   Reported over: missing, `band` (the seam, +/- `--rings` hops), overall.

Why (2) is not redundant: DDIM clamping makes known vertices exact BY
CONSTRUCTION, so a crease at the seam has zero positional error and is invisible
to (1) -- the `band` columns are the ones that see it. And an MSE regressor wins
on (1) precisely by predicting the over-smoothed conditional mean, which should
COST it on (2). See utils/normals.py for the full argument.

Works with the reorganized test_set layout where GT and masks live in separate
subfolders. Defaults match:
    gt   : data/completion_test_samples/test_set/gt/<name>_gt_complete.ply
    mask : data/completion_test_samples/test_set/partial/<name>_mask.npy
    arm  : <dir>/<name>_completed_diffusion.ply

Arm filename suffixes are AUTO-DETECTED, so non-diffusion arms score without
extra flags (the regressor writes `<name>_completed_regressor.ply`). Override
explicitly with a third field: `label:dir:_completed_regressor.ply`.

Usage (compare eps vs x0 vs regressor):
  python scripts/compare_arms.py \
      --arm "eps:data/completion_test_samples/test_set/completed_diffusion_ ε" \
      --arm  "x0:data/completion_test_samples/test_set/completed_diffusion_x0" \
      --arm "regressor:data/completion_test_samples/test_set/temp/completed_regressor"

Outputs (written next to the first arm's parent, i.e. test_set/):
  - arm_comparison_per_sample.csv   (one row per sample x arm)
  - arm_comparison_summary.csv      (one row per arm: aggregate mean +/- std)
  and prints a summary table.
"""
import os, sys, argparse, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from glob import glob
from trimesh import load_mesh

from utils.normals import normal_angular_error, boundary_band, normal_stats
from utils.boundary_field import BoundaryFieldSolver

DEF_TEST_DIR = os.path.join('data', 'completion_test_samples', 'test_set')
DEF_SUFFIX = '_completed_diffusion.ply'


def region_stats(err, sel, prefix):
    """mean/median/max/std of per-vertex error over a boolean-selected region."""
    if sel.any():
        e = err[sel]
        return {f'{prefix}_mean': float(e.mean()),
                f'{prefix}_median': float(np.median(e)),
                f'{prefix}_max': float(e.max()),
                f'{prefix}_std': float(e.std())}
    return {f'{prefix}_mean': 0.0, f'{prefix}_median': 0.0,
            f'{prefix}_max': 0.0, f'{prefix}_std': 0.0}


def detect_suffix(comp_dir, name):
    """
    Work out how this arm names its files, so non-diffusion arms need no flag.

    Prefers the diffusion default when present, otherwise falls back to the
    unique `<name>_*.ply` in the folder. Ambiguity is reported rather than
    guessed at -- silently scoring the wrong file would be worse than failing.
    """
    if os.path.exists(os.path.join(comp_dir, name + DEF_SUFFIX)):
        return DEF_SUFFIX
    cands = sorted(glob(os.path.join(comp_dir, f'{name}_*.ply')))
    if len(cands) == 1:
        return os.path.basename(cands[0])[len(name):]
    if len(cands) > 1:
        opts = ', '.join(os.path.basename(c)[len(name):] for c in cands)
        print(f"  ! {comp_dir}: ambiguous suffix for {name} ({opts}); "
              f"pass label:dir:suffix to disambiguate")
    return None


def eval_arm(label, comp_dir, gt_dir, mask_dir, faces, adjacency, rings,
             suffix=None):
    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    rows = []
    for gt_file in gt_files:
        name = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{name}_mask.npy')
        sfx = suffix or detect_suffix(comp_dir, name)
        comp_file = os.path.join(comp_dir, f'{name}{sfx}') if sfx else ''
        if not (os.path.exists(mask_file) and comp_file and os.path.exists(comp_file)):
            print(f"  [{label}] skip {name}: missing mask or completed file")
            continue

        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        comp = np.array(load_mesh(comp_file, process=False).vertices, dtype=np.float32)
        mask = np.load(mask_file)                       # True = missing
        if not (len(gt) == len(comp) == len(mask)):
            print(f"  [{label}] skip {name}: size mismatch "
                  f"(gt={len(gt)} comp={len(comp)} mask={len(mask)})")
            continue

        err = np.linalg.norm(gt - comp, axis=1)         # per-vertex L2 (meters)
        # bbox diagonal of the GT (meters) -> scale-invariant % error, comparable
        # across datasets of different physical size (COMA heads vs DFAUST bodies).
        diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
        row = {'arm': label, 'sample': name,
               'missing_pct': 100.0 * mask.sum() / len(mask),
               'bbox_diag_m': diag}
        row.update(region_stats(err, mask, 'missing'))
        row.update(region_stats(err, np.ones_like(mask), 'overall'))
        row.update(region_stats(err, ~mask, 'known'))
        # Interpretable duplicates of the headline means: millimeters + % of diagonal.
        row['missing_mm'] = 1000.0 * row['missing_mean']
        row['overall_mm'] = 1000.0 * row['overall_mean']
        row['missing_pct_diag'] = 100.0 * row['missing_mean'] / diag if diag else 0.0
        row['overall_pct_diag'] = 100.0 * row['overall_mean'] / diag if diag else 0.0

        # --- orientation: what the positional metric structurally cannot see ---
        ang, valid = normal_angular_error(comp, gt, faces)
        band = boundary_band(adjacency, mask, rings=rings)
        row.update(normal_stats(ang, valid, mask, 'nrm_missing'))
        row.update(normal_stats(ang, valid, band, 'nrm_band'))
        # Cleanest crease isolate: the KNOWN side of the band only. Those
        # vertices are composited back from the input, so their positions are
        # exact for every arm -- any normal error there is inherited purely from
        # their predicted neighbours, i.e. it IS the crease and nothing else.
        # (`band` alone mixes exact and predicted vertices, which dilutes it.)
        row.update(normal_stats(ang, valid, band & ~mask, 'nrm_seam'))
        row.update(normal_stats(ang, valid, np.ones_like(mask), 'nrm_overall'))

        rows.append(row)
        print(f"  [{label}] {name}: missing={row['missing_mm']:.3f} mm  "
              f"overall={row['overall_mm']:.3f} mm  |  normals: "
              f"missing={row['nrm_missing_deg_mean']:.2f} deg  "
              f"band={row['nrm_band_deg_mean']:.2f} deg  "
              f"({row['missing_pct']:.1f}% missing)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True,
                    help='label:dir[:suffix]  (repeatable). suffix auto-detected if omitted')
    ap.add_argument('--gt_dir', default=os.path.join(DEF_TEST_DIR, 'gt'))
    ap.add_argument('--mask_dir', default=os.path.join(DEF_TEST_DIR, 'partial'))
    ap.add_argument('--out_dir', default=DEF_TEST_DIR)
    ap.add_argument('--rings', type=int, default=2,
                    help='width of the seam band (hops each side) for normal metrics')
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        # Windows paths contain ':' (D:\...), so split from the LEFT for the
        # label and only treat a trailing '.ply' field as an explicit suffix.
        if ':' not in spec:
            ap.error(f"--arm must be label:dir[:suffix], got {spec!r}")
        label, rest = spec.split(':', 1)
        suffix = None
        if rest.rstrip().endswith('.ply') and ':' in rest:
            rest, suffix = rest.rsplit(':', 1)
            suffix = suffix.strip()
        arms.append((label.strip(), rest.strip(), suffix))

    # Topology is fixed and shared by every arm, so faces + adjacency are built
    # once from the first GT mesh rather than per sample.
    gt_files = sorted(glob(os.path.join(args.gt_dir, '*_gt_complete.ply')))
    if not gt_files:
        print(f"No GT meshes in {args.gt_dir} -- check --gt_dir."); return
    tpl = load_mesh(gt_files[0], process=False)
    faces = np.asarray(tpl.faces, dtype=np.int64)
    adjacency = BoundaryFieldSolver(faces, np.asarray(tpl.vertices))._A

    print(f"GT dir  : {args.gt_dir}")
    print(f"Mask dir: {args.mask_dir}")
    print(f"Topology: {len(tpl.vertices)} verts / {len(faces)} faces | "
          f"seam band = {args.rings} rings each side\n")

    all_rows, summary = [], []
    for label, comp_dir, suffix in arms:
        print(f"=== arm '{label}'  <-  {comp_dir} ===")
        rows = eval_arm(label, comp_dir, args.gt_dir, args.mask_dir,
                        faces, adjacency, args.rings, suffix)
        all_rows.extend(rows)
        if not rows:
            print(f"  (no samples scored for {label})\n"); continue
        agg = {'arm': label, 'n': len(rows)}
        for key in ['missing_mean', 'overall_mean', 'known_mean',
                    'missing_median', 'missing_max',
                    'missing_mm', 'overall_mm',
                    'missing_pct_diag', 'overall_pct_diag',
                    'nrm_missing_deg_mean', 'nrm_missing_deg_median',
                    'nrm_missing_deg_p95', 'nrm_missing_nc',
                    'nrm_band_deg_mean', 'nrm_band_deg_p95', 'nrm_band_nc',
                    'nrm_seam_deg_mean', 'nrm_seam_deg_p95', 'nrm_seam_nc',
                    'nrm_overall_deg_mean', 'nrm_overall_nc']:
            vals = np.array([r[key] for r in rows])
            agg[key] = float(vals.mean())
            agg[key + '_std'] = float(vals.std())
        summary.append(agg)
        print()

    if not all_rows:
        print("Nothing scored -- check paths."); return

    # per-sample CSV
    per_path = os.path.join(args.out_dir, 'arm_comparison_per_sample.csv')
    fields = list(all_rows[0].keys())
    with open(per_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)

    # summary CSV
    sum_path = os.path.join(args.out_dir, 'arm_comparison_summary.csv')
    sfields = list(summary[0].keys())
    with open(sum_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sfields); w.writeheader(); w.writerows(summary)

    # printed summary tables -- position first, then orientation
    w = 100
    print("=" * w)
    print("POSITION -- per-vertex L2 to GT (mean over samples; lower is better)")
    print("-" * w)
    print(f"{'arm':<14} {'n':>3}  {'missing (m)':>13}  {'missing (mm)':>13}  "
          f"{'% bbox diag':>13}  {'overall (mm)':>13}")
    for a in summary:
        print(f"{a['arm']:<14} {a['n']:>3}  "
              f"{a['missing_mean']:>13.6f}  {a['missing_mm']:>13.3f}  "
              f"{a['missing_pct_diag']:>12.3f}%  {a['overall_mm']:>13.3f}")

    print("\n" + "=" * w)
    print("ORIENTATION -- vertex-normal angular error in DEGREES (lower is better)")
    print(f"  'seam' = KNOWN side of the boundary only. Positions there are exact for every\n"
          f"  arm, so its normal error is inherited purely from predicted neighbours = crease.")
    print("-" * w)
    print(f"{'arm':<14} {'n':>3}  {'missing deg':>12}  {'p95':>7}  "
          f"{'band deg':>9}  {'seam deg':>9}  {'seam p95':>9}  {'overall':>8}  {'NC(miss)':>9}")
    for a in summary:
        print(f"{a['arm']:<14} {a['n']:>3}  "
              f"{a['nrm_missing_deg_mean']:>12.3f}  {a['nrm_missing_deg_p95']:>7.2f}  "
              f"{a['nrm_band_deg_mean']:>9.3f}  {a['nrm_seam_deg_mean']:>9.3f}  "
              f"{a['nrm_seam_deg_p95']:>9.2f}  "
              f"{a['nrm_overall_deg_mean']:>8.3f}  {a['nrm_missing_nc']:>9.4f}")
    print("=" * w)

    # Rank on both axes. Disagreement between them is the interesting outcome:
    # it means the arms trade position against surface quality, which is exactly
    # the regressor-vs-diffusion hypothesis this metric was added to test.
    if len(summary) >= 2:
        best_pos = min(summary, key=lambda a: a['missing_mean'])
        best_nrm = min(summary, key=lambda a: a['nrm_missing_deg_mean'])
        best_band = min(summary, key=lambda a: a['nrm_band_deg_mean'])
        print(f"best POSITION (missing) : {best_pos['arm']} "
              f"({best_pos['missing_mm']:.3f} mm)")
        print(f"best ORIENTATION (missing): {best_nrm['arm']} "
              f"({best_nrm['nrm_missing_deg_mean']:.3f} deg)")
        print(f"best ORIENTATION (band) : {best_band['arm']} "
              f"({best_band['nrm_band_deg_mean']:.3f} deg)")
        if best_pos['arm'] != best_nrm['arm']:
            print("-> the two axes DISAGREE: the most accurate arm is not the "
                  "one with the best surface.")
    print(f"\nwrote {per_path}\nwrote {sum_path}")


if __name__ == '__main__':
    main()
