"""
Break any method's completed meshes down by masking strategy.

Table 1 of the paper reports per-strategy numbers (patch / plane cut / scatter)
because the aggregate hides the finding that difficulty tracks geodesic depth
rather than how much surface is missing. Baselines need the same breakdown or
they cannot be compared row for row.

Usage
  python breakdown_by_mask.py --completed_dir 3DGeoMeshNet/completed_fit_oracle \
      --suffix _completed_3dgeomeshnet.ply --method 3DGeoMeshNet
  python breakdown_by_mask.py --completed_dir <dir> --suffix <s> --method <m> \
      --out_csv breakdown.csv
"""

import os
import csv
import argparse
from glob import glob

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DIS_ROOT = os.path.abspath(os.path.join(HERE, ".."))
TEST_DIR = os.path.join(DIS_ROOT, "data", "completion_test_samples", "test_set")

TYPES = ("bfs_patch", "plane_cut", "random_scatter")


def mask_type(partial_dir, base):
    hits = glob(os.path.join(partial_dir, f"{base}_partial_*.ply"))
    if not hits:
        return "unknown"
    name = os.path.basename(hits[0])
    for t in TYPES:
        if t in name:
            return t
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completed_dir", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--method", default="method")
    ap.add_argument("--test_dir", default=TEST_DIR)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    gt_dir = os.path.join(args.test_dir, "gt")
    gt_dir = gt_dir if os.path.isdir(gt_dir) else args.test_dir
    partial_dir = os.path.join(args.test_dir, "partial")
    partial_dir = partial_dir if os.path.isdir(partial_dir) else args.test_dir

    rows = []
    for gt_file in sorted(glob(os.path.join(gt_dir, "*_gt_complete.ply"))):
        base = os.path.basename(gt_file).replace("_gt_complete.ply", "")
        mask_file = os.path.join(partial_dir, f"{base}_mask.npy")
        cand = (os.path.join(args.completed_dir, f"{base}{args.suffix}") if args.suffix
                else next(iter(sorted(glob(os.path.join(args.completed_dir, f"{base}*.ply")))), None))
        if not (cand and os.path.exists(cand) and os.path.exists(mask_file)):
            continue

        gt = np.asarray(trimesh.load(gt_file, process=False).vertices, dtype=np.float64)
        pr = np.asarray(trimesh.load(cand, process=False).vertices, dtype=np.float64)
        if gt.shape != pr.shape:
            print(f"  skip {base}: {pr.shape} != {gt.shape}")
            continue
        m = np.load(mask_file).astype(bool)
        e = np.linalg.norm(gt - pr, axis=1)
        rows.append({"sample": base, "type": mask_type(partial_dir, base),
                     "missing_pct": 100.0 * m.mean(),
                     "hole_mm": 1000.0 * e[m].mean(),
                     "full_mm": 1000.0 * e.mean(),
                     "known_mm": 1000.0 * e[~m].mean()})

    if not rows:
        raise SystemExit(f"nothing scored in {args.completed_dir}")

    print(f"\n{args.method}  ({len(rows)} meshes)")
    print(f"{'type':<16} {'n':>3} {'miss%':>7} {'hole mm':>10} {'full mm':>9} {'known mm':>10}")
    print("-" * 60)
    for t in TYPES + ("unknown",):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        print(f"{t:<16} {len(sub):>3} "
              f"{np.mean([r['missing_pct'] for r in sub]):>7.1f} "
              f"{np.mean([r['hole_mm'] for r in sub]):>10.3f} "
              f"{np.mean([r['full_mm'] for r in sub]):>9.3f} "
              f"{np.mean([r['known_mm'] for r in sub]):>10.3f}")
    print("-" * 60)
    print(f"{'ALL':<16} {len(rows):>3} "
          f"{np.mean([r['missing_pct'] for r in rows]):>7.1f} "
          f"{np.mean([r['hole_mm'] for r in rows]):>10.3f} "
          f"{np.mean([r['full_mm'] for r in rows]):>9.3f} "
          f"{np.mean([r['known_mm'] for r in rows]):>10.3f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-sample -> {args.out_csv}")


if __name__ == "__main__":
    main()
