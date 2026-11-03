"""
Stage the `Test_old` real cropped partials into the standard eval layout.

Test_old is stored differently from the synthetic held-out set: the partial
meshes have their missing vertices genuinely DELETED (fewer vertices, e.g.
no_half has 2568 of the template's 5023), rather than zeroed in place. Nothing
downstream can consume that directly -- correspondence is lost, so there is no
mask and no vertex-to-vertex comparison.

The mask is recovered the same way scripts/test_old_meshes.py does it: for each
GT vertex, find the nearest vertex in the partial; if nothing lands within
`--threshold` the vertex was cropped away. GT is already in template topology
(5023 verts), so the recovered mask is a valid (N,) bool over the template.

Writes, for each pair, into <out>/gt and <out>/partial:
    <name>_gt_complete.ply     (the GT mesh, unchanged)
    <name>_mask.npy            (N, bool -- True = missing)
so scripts/ablation_consistency.py and friends read it with no special-casing.

Run:
  python scripts/stage_test_old.py
  python scripts/stage_test_old.py --out <dir> --threshold 0.01
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import shutil
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# GT filename -> partial filename. Note gt_plain/no_plane: the two files really
# are spelled differently on disk, so this cannot be derived by substitution.
PAIRS = [
    ("gt_ear.ply",   "no_ear.ply"),
    ("gt_eyes.ply",  "no_eyes.ply"),
    ("gt_half.ply",  "no_half.ply"),
    ("gt_half2.ply", "no_half2.ply"),
    ("gt_mouth.ply", "no_mouth.ply"),
    ("gt_nose.ply",  "no_nose.ply"),
    ("gt_plain.ply", "no_plane.ply"),
]

DEFAULT_ROOT = os.path.join('data', 'completion_test_samples', 'test_set', 'Test_old')


def recover_mask(gt_verts, partial_verts, threshold):
    """True where the GT vertex has no counterpart in the cropped partial."""
    dist, _ = cKDTree(partial_verts).query(gt_verts)
    return dist > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--out', default=None, help='default: <root>/staged')
    ap.add_argument('--threshold', type=float, default=1e-6,
                    help="max distance for a GT vertex to count as observed. The "
                         "partials are exact vertex SUBSETS of the GT, so an exact "
                         "match is what is wanted. Do not loosen this: the template's "
                         "median edge is 0.0031, so a threshold anywhere near that "
                         "lets a deleted vertex match its own neighbour and be "
                         "counted as observed (at 0.01, no_nose recovers as 0%% "
                         "missing instead of 0.6%%).")
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, 'staged')
    gt_out = os.path.join(out, 'gt')
    mask_out = os.path.join(out, 'partial')
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    print(f'{"name":10s} {"gt":>6s} {"partial":>8s} {"missing":>8s} {"missing %":>10s}')
    print('-' * 46)
    rows = []
    for gt_name, part_name in PAIRS:
        gt_path = os.path.join(args.root, 'gt', gt_name)
        part_path = os.path.join(args.root, 'partial', part_name)
        if not (os.path.exists(gt_path) and os.path.exists(part_path)):
            print(f'  ! missing pair {gt_name}/{part_name}, skipped')
            continue

        gt_mesh = trimesh.load(gt_path, process=False)
        part_mesh = trimesh.load(part_path, process=False)
        gt_v = np.asarray(gt_mesh.vertices, dtype=np.float64)
        part_v = np.asarray(part_mesh.vertices, dtype=np.float64)

        mask = recover_mask(gt_v, part_v, args.threshold)
        # Nothing was moved, only deleted, so exactly (N - n_partial) GT vertices
        # must fail to match. If this does not hold the threshold is wrong and
        # every downstream number would be computed against the wrong region.
        expected = len(gt_v) - len(part_v)
        if int(mask.sum()) != expected:
            raise SystemExit(
                f'{part_name}: recovered {int(mask.sum())} missing vertices but the '
                f'partial is {expected} vertices short of the GT. threshold='
                f'{args.threshold:g} is mismatched for this data.')
        base = os.path.splitext(part_name)[0]          # e.g. 'no_half'

        shutil.copyfile(gt_path, os.path.join(gt_out, f'{base}_gt_complete.ply'))
        np.save(os.path.join(mask_out, f'{base}_mask.npy'), mask)

        pct = 100.0 * mask.sum() / len(mask)
        rows.append((base, pct))
        print(f'{base:10s} {len(gt_v):6d} {len(part_v):8d} {int(mask.sum()):8d} '
              f'{pct:9.2f}%')

    # The recovered counts should land on the published hole sizes; a threshold
    # that is too loose or too tight shows up here as a shifted percentage
    # rather than as a silent change in what is being measured.
    print(f'\nstaged {len(rows)} pairs -> {out}')
    print('expected (paper Table: robustness): 0.3 / 0.6 / 6.0 / 11.7 / 38.8 / '
          '46.9 / 48.9 %')
    print('recovered: ' + ' / '.join(f'{p:.1f}' for _, p in sorted(rows, key=lambda r: r[1])))


if __name__ == '__main__':
    main()
