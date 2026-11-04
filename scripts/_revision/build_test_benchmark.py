"""
Build the fixed completion benchmark on the official COMA test set.

Every test mesh gets one mask per strategy (random_scatter, bfs_patch,
plane_cut), so the three mask types are perfectly balanced. Hole sizes are drawn
uniformly from [ratio_min, ratio_max] with a fixed seed, so results can later be
stratified by hole size. The SAME masks must be used for every method/baseline.

Output (revision/benchmark/):
    official_test_masks.npz   masks (M, N) bool, True = missing
    official_test_index.csv   row, subject, expression, frame, gt_path, strategy,
                              target_ratio, actual_ratio

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/build_test_benchmark.py
"""
import os
import sys
import csv
import argparse
from glob import glob

import numpy as np
from trimesh import load_mesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from utils.masking import MASK_STRATEGIES, _build_adjacency

STRATEGIES = ('random_scatter', 'bfs_patch', 'plane_cut')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_root', default=os.path.join(_ROOT, 'data', 'COMA_test_subset', 'gt'))
    ap.add_argument('--out_dir', default=os.path.join(_ROOT, 'revision', 'benchmark'))
    ap.add_argument('--ratio_min', type=float, default=0.05)
    ap.add_argument('--ratio_max', type=float, default=0.40)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.gt_root, '*', '*.ply')))
    if not files:
        raise SystemExit(f'No test meshes under {args.gt_root}')
    os.makedirs(args.out_dir, exist_ok=True)

    first = load_mesh(files[0], process=False)
    faces = np.asarray(first.faces)
    n_verts = len(first.vertices)
    adj = _build_adjacency(faces, n_verts)
    rng = np.random.RandomState(args.seed)

    masks, rows = [], []
    for f in files:
        verts = np.asarray(load_mesh(f, process=False).vertices, dtype=np.float32)
        assert len(verts) == n_verts, f
        subject = os.path.basename(os.path.dirname(f))
        expression, frame = os.path.basename(f).split('.')[:2]
        for strat in STRATEGIES:
            ratio = rng.uniform(args.ratio_min, args.ratio_max)
            m = MASK_STRATEGIES[strat](verts, faces, mask_ratio=ratio, rng=rng, adj=adj)
            masks.append(np.asarray(m, dtype=bool))
            rows.append(dict(row=len(rows), subject=subject, expression=expression,
                             frame=frame, gt_path=os.path.relpath(f, _ROOT),
                             strategy=strat, target_ratio=round(ratio, 4),
                             actual_ratio=round(float(m.mean()), 4)))

    np.savez_compressed(os.path.join(args.out_dir, 'official_test_masks.npz'),
                        masks=np.stack(masks))
    with open(os.path.join(args.out_dir, 'official_test_index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ratios = np.array([r['actual_ratio'] for r in rows])
    print(f'{len(files)} meshes x {len(STRATEGIES)} strategies = {len(rows)} samples')
    for s in STRATEGIES:
        r = ratios[[x['strategy'] == s for x in rows]]
        print(f'  {s:15s} n={len(r)}  missing {100*r.mean():.1f}% '
              f'(min {100*r.min():.1f}, max {100*r.max():.1f})')
    print('saved to', args.out_dir)


if __name__ == '__main__':
    main()
