"""
Generic, apples-to-apples evaluator for ANY completion method.

Given the fixed test set (GT complete + mask) and a directory of completed meshes
in the *template topology* (5023 verts, same vertex order), it computes the same
per-vertex L2 metrics used for our diffusion model:
    - missing-region error  = mean ||gt - pred|| over masked (missing) vertices
    - overall error         = mean over all vertices
    - known-region error    = mean over observed vertices
and prints an aggregate (mean +/- std over samples) plus a one-line summary row
suitable for pasting into the paper. Optionally appends/writes a CSV.

This is the harness that makes every baseline comparable to Table 1 in the paper.

Usage:
  python research/eval_completion_dir.py \
      --completed_dir data/completion_test_samples/test_set/completed_diffusion \
      --method diffusion

  # match a specific filename pattern after the basename:
  python research/eval_completion_dir.py \
      --completed_dir <dir> --suffix _completed_laplacian.ply --method laplacian
"""

import os
import sys
import csv
import argparse
from glob import glob

import numpy as np
import trimesh

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, 'data',
                                'completion_test_samples', 'test_set')


def load_verts(path):
    m = trimesh.load(path, process=False)
    return np.asarray(m.vertices, dtype=np.float64)


def find_completed(completed_dir, basename, suffix):
    """Locate the completed mesh for a given test basename."""
    if suffix:
        cand = os.path.join(completed_dir, f'{basename}{suffix}')
        return cand if os.path.exists(cand) else None
    hits = sorted(glob(os.path.join(completed_dir, f'{basename}*.ply')))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', default=DEFAULT_TEST_DIR)
    ap.add_argument('--gt_dir', default=None,
                    help="dir holding *_gt_complete.ply; defaults to <test_dir>/gt "
                         "if that exists, else <test_dir>")
    ap.add_argument('--mask_dir', default=None,
                    help="dir holding *_mask.npy; defaults to <test_dir>/partial "
                         "if that exists, else <test_dir>")
    ap.add_argument('--completed_dir', required=True,
                    help='dir with completed meshes in template topology')
    ap.add_argument('--suffix', default='',
                    help="filename suffix after basename, e.g. '_completed_mean.ply'."
                         " If empty, first '{basename}*.ply' match is used.")
    ap.add_argument('--method', default='method', help='label for the CSV / summary')
    ap.add_argument('--out_csv', default=None,
                    help='if set, append a one-row aggregate to this CSV')
    ap.add_argument('--per_sample_csv', default=None,
                    help='if set, write the full per-sample table here')
    args = ap.parse_args()

    # The test set may be flat, or split into gt/ and partial/ subdirectories.
    gt_dir = args.gt_dir
    if gt_dir is None:
        cand = os.path.join(args.test_dir, 'gt')
        gt_dir = cand if os.path.isdir(cand) else args.test_dir
    mask_dir = args.mask_dir
    if mask_dir is None:
        cand = os.path.join(args.test_dir, 'partial')
        mask_dir = cand if os.path.isdir(cand) else args.test_dir

    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if not gt_files:
        sys.exit(f'No *_gt_complete.ply under {gt_dir}')

    rows = []
    for gt_file in gt_files:
        basename = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{basename}_mask.npy')
        if not os.path.exists(mask_file):
            print(f'  skip {basename}: no mask'); continue

        pred_file = find_completed(args.completed_dir, basename, args.suffix)
        if pred_file is None:
            print(f'  skip {basename}: no completed mesh in {args.completed_dir}')
            continue

        gt = load_verts(gt_file)
        pred = load_verts(pred_file)
        mask = np.load(mask_file).astype(bool)  # True = missing

        if gt.shape != pred.shape:
            print(f'  skip {basename}: shape {pred.shape} != GT {gt.shape} '
                  f'(not in template topology?)'); continue

        err = np.linalg.norm(gt - pred, axis=1)
        rows.append({
            'sample': basename,
            'missing_pct': 100.0 * mask.sum() / len(mask),
            'missing_mean': float(err[mask].mean()) if mask.any() else 0.0,
            'overall_mean': float(err.mean()),
            'known_mean': float(err[~mask].mean()) if (~mask).any() else 0.0,
            'missing_max': float(err[mask].max()) if mask.any() else 0.0,
        })

    if not rows:
        sys.exit('No samples scored.')

    miss = np.array([r['missing_mean'] for r in rows])
    over = np.array([r['overall_mean'] for r in rows])
    known = np.array([r['known_mean'] for r in rows])
    mpct = np.array([r['missing_pct'] for r in rows])

    print('\n' + '=' * 70)
    print(f'METHOD: {args.method}   ({len(rows)} samples)')
    print('=' * 70)
    print(f'  avg missing %        : {mpct.mean():.1f}%')
    print(f'  missing-region error : {miss.mean():.6f} +/- {miss.std():.6f}')
    print(f'  overall error        : {over.mean():.6f} +/- {over.std():.6f}')
    print(f'  known-region error   : {known.mean():.6f}')
    print(f'\n  >> paper row:  {args.method} & {miss.mean():.5f} & {over.mean():.5f} '
          f'& {known.mean():.5f} \\\\')

    if args.per_sample_csv:
        with open(args.per_sample_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'  per-sample CSV: {args.per_sample_csv}')

    if args.out_csv:
        header = ['method', 'n', 'avg_missing_pct',
                  'missing_mean', 'missing_std',
                  'overall_mean', 'overall_std', 'known_mean']
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow([args.method, len(rows), f'{mpct.mean():.2f}',
                        f'{miss.mean():.6f}', f'{miss.std():.6f}',
                        f'{over.mean():.6f}', f'{over.std():.6f}',
                        f'{known.mean():.6f}'])
        print(f'  appended aggregate to: {args.out_csv}')


if __name__ == '__main__':
    main()
