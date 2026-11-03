"""
Classical, dependency-light completion baselines (runnable now, no external repo).

These give honest non-learned comparators for Table 1, scored by the SAME metric
as the diffusion model (see eval_completion_dir.py):

  * mean       : fill missing vertices with the template (mean-shape) position.
                 The trivial "predict the average face" lower bound.
  * laplacian  : harmonic / membrane infill. Solve the uniform graph-Laplacian
                 system  L_mm X_m = -L_mk X_k  with the observed (known) vertices
                 as Dirichlet boundary conditions -> the smoothest surface that
                 interpolates the hole. The standard classical mesh hole-filling
                 baseline.

Both keep the observed vertices exactly (known-region error = 0), matching the
protocol used for our method, so the comparison is fair.

Writes completed meshes (template topology) to:
  <test_dir>/completed_<method>/<basename>_completed_<method>.ply

Usage:
  python scripts/baselines_classical.py --method mean
  python scripts/baselines_classical.py --method laplacian
Then score with:
  python scripts/eval_completion_dir.py \
     --completed_dir data/completion_test_samples/test_set/completed_laplacian \
     --suffix _completed_laplacian.ply --method laplacian \
     --out_csv data/completion_test_samples/baseline_results.csv
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import trimesh
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, 'data',
                                'completion_test_samples', 'test_set')
TEMPLATE_PLY = os.path.join(PROJECT_ROOT, 'data', 'template.ply')


def uniform_laplacian(n, faces):
    """Combinatorial graph Laplacian L = D - A from triangle faces."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])               # undirected
    e = np.unique(e, axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    return L.tocsr()


def fill_mean(partial, mask, template):
    out = partial.copy()
    out[mask] = template[mask]
    return out


def fill_laplacian(partial, mask, template, L):
    """Membrane infill: minimal-energy surface through the known boundary."""
    out = partial.copy()
    known = ~mask
    miss = mask
    if not miss.any():
        return out
    Lmm = L[miss][:, miss]
    Lmk = L[miss][:, known]
    rhs = -(Lmk @ partial[known])                # (n_miss, 3)
    try:
        out[miss] = spsolve(Lmm.tocsc(), rhs)
    except Exception as ex:
        print(f'    laplacian solve failed ({ex}); falling back to mean fill')
        out[miss] = template[miss]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['mean', 'laplacian'], required=True)
    ap.add_argument('--test_dir', default=DEFAULT_TEST_DIR)
    ap.add_argument('--template', default=TEMPLATE_PLY)
    args = ap.parse_args()

    tmpl = trimesh.load(args.template, process=False)
    faces = np.asarray(tmpl.faces)
    template = np.asarray(tmpl.vertices, dtype=np.float64)
    n = len(template)

    L = uniform_laplacian(n, faces) if args.method == 'laplacian' else None

    out_dir = os.path.join(args.test_dir, f'completed_{args.method}')
    os.makedirs(out_dir, exist_ok=True)

    partial_files = sorted(glob(os.path.join(args.test_dir, '*_partial_*.ply')))
    if not partial_files:
        sys.exit(f'No *_partial_*.ply under {args.test_dir}')

    for pf in partial_files:
        base = os.path.basename(pf)
        basename = base.split('_partial_')[0]
        mask_file = os.path.join(args.test_dir, f'{basename}_mask.npy')
        if not os.path.exists(mask_file):
            print(f'  skip {basename}: no mask'); continue

        partial = np.asarray(trimesh.load(pf, process=False).vertices, dtype=np.float64)
        mask = np.load(mask_file).astype(bool)    # True = missing
        if partial.shape[0] != n:
            print(f'  skip {basename}: {partial.shape[0]} verts != template {n}')
            continue

        if args.method == 'mean':
            comp = fill_mean(partial, mask, template)
        else:
            comp = fill_laplacian(partial, mask, template, L)

        out_path = os.path.join(out_dir, f'{basename}_completed_{args.method}.ply')
        trimesh.Trimesh(vertices=comp, faces=faces, process=False).export(out_path)
        print(f'  {basename}: missing {int(mask.sum())} -> {out_path}')

    print(f'\nDone. Completed meshes in: {out_dir}')
    print('Score with scripts/eval_completion_dir.py (see header).')


if __name__ == '__main__':
    main()
