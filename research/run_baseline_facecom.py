"""
Baseline adapter: FaceCom (CVPR 2024) -> our template topology.

FaceCom completes a partial face by OPTIMIZING its mesh-generator's latent code
and a rigid transform against the partial input (image-inpainting guided). It is
our closest competitor and the must-beat baseline: same correspondence-mesh
setting, but slow test-time optimization, soft constraint, single deterministic
output.

This adapter runs FaceCom per partial and remaps its output onto OUR template
topology so it is scorable by eval_completion_dir.py:

  1. export the observed (known) vertices of each partial as the FaceCom input
     (FaceCom accepts irregular partial meshes / point sets).
  2. run FaceCom optimization -> a complete face mesh in FaceCom's OWN topology.
  3. remap to our COMA template (5023 verts): for each MISSING template vertex,
     take the nearest point on the FaceCom output surface (trimesh proximity);
     copy KNOWN vertices verbatim from the partial (fair protocol, known err = 0).
  4. save <test_dir>/completed_facecom/<basename>_completed_facecom.ply

Because FaceCom optimizes a rigid transform internally, its output may be in a
different pose; step 3's nearest-surface remap is pose-tolerant but you should
sanity-check one sample visually before trusting the numbers.

SETUP (run once):
  git clone https://github.com/dragonylee/FaceCom
  # install its deps and download its pretrained generator per their README.
  set FACECOM_ROOT=C:\path\to\FaceCom

Then (FaceCom typically exposes a completion entry point; adjust CALL below to
match their API), and finally:
  python research/run_baseline_facecom.py
  python research/eval_completion_dir.py \
     --completed_dir data/completion_test_samples/test_set/completed_facecom \
     --suffix _completed_facecom.ply --method FaceCom \
     --out_csv data/completion_test_samples/baseline_results.csv
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import trimesh

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, 'data',
                                'completion_test_samples', 'test_set')


def run_facecom_on(known_pts, facecom_root):
    """
    Call FaceCom's completion on a partial point set -> complete trimesh.

    FaceCom's public code exposes an optimization routine that fits its generator
    to an input partial mesh. Wire the call below to their actual API (the repo's
    README / demo script shows the exact entry point and arguments). The function
    must return a trimesh.Trimesh (FaceCom's own topology).
    """
    if not facecom_root or not os.path.isdir(facecom_root):
        sys.exit('Set FACECOM_ROOT to your cloned FaceCom repo (see header).')
    sys.path.insert(0, facecom_root)
    try:
        # >>> ADJUST to FaceCom's real API (see their demo/optimization script) <<<
        # from completion import complete_partial   # example name
        # mesh = complete_partial(known_pts)         # returns verts/faces
        raise NotImplementedError(
            'Wire run_facecom_on() to FaceCom\'s completion entry point. '
            'See the FaceCom repo demo script for the exact function + args.')
    except NotImplementedError as ex:
        sys.exit(str(ex))


def remap_to_template(facecom_mesh, partial, mask):
    """Missing template verts <- nearest point on FaceCom surface; keep known."""
    comp = partial.copy()
    closest, _, _ = trimesh.proximity.closest_point(facecom_mesh, partial[mask])
    comp[mask] = closest
    comp[~mask] = partial[~mask]
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', default=DEFAULT_TEST_DIR)
    ap.add_argument('--facecom_root', default=os.environ.get('FACECOM_ROOT'))
    args = ap.parse_args()

    # faces from any GT (template topology) for export
    any_gt = sorted(glob(os.path.join(args.test_dir, '*_gt_complete.ply')))[0]
    faces = np.asarray(trimesh.load(any_gt, process=False).faces)

    out_dir = os.path.join(args.test_dir, 'completed_facecom')
    os.makedirs(out_dir, exist_ok=True)

    for pf in sorted(glob(os.path.join(args.test_dir, '*_partial_*.ply'))):
        basename = os.path.basename(pf).split('_partial_')[0]
        mask_file = os.path.join(args.test_dir, f'{basename}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        partial = np.asarray(trimesh.load(pf, process=False).vertices, dtype=np.float64)
        mask = np.load(mask_file).astype(bool)

        fc_mesh = run_facecom_on(partial[~mask], args.facecom_root)
        comp = remap_to_template(fc_mesh, partial, mask)

        out_path = os.path.join(out_dir, f'{basename}_completed_facecom.ply')
        trimesh.Trimesh(vertices=comp, faces=faces, process=False).export(out_path)
        print(f'  {basename}: -> {out_path}')

    print(f'\nDone. See {out_dir}. Score with eval_completion_dir.py.')


if __name__ == '__main__':
    main()
