"""
Baseline adapter: AdaPoinTr (point-cloud completion) -> our template topology.

AdaPoinTr is an EXTERNAL repo with its own deps (PoinTr / PointTransformer).
This adapter does the plumbing so its output is scorable by eval_completion_dir.py:

  1. partial mesh  -> input point cloud = the OBSERVED (known) vertices.
  2. AdaPoinTr inference -> dense completed point cloud (unordered, no
     correspondence to our template).
  3. remap to template topology: for each MISSING template vertex, take the
     nearest predicted point; copy the KNOWN vertices verbatim from the partial
     (same protocol we give our own model -> fair, known-region error = 0).
  4. save <test_dir>/completed_adapointr/<basename>_completed_adapointr.ply

NOTE ON FAIRNESS: AdaPoinTr cannot preserve the observed region exactly nor
report per-vertex uncertainty (points are unordered); step 3 grants it the known
region for free, which is the most generous fair protocol. The missing-region
error it gets is the number to report.

SETUP (run once, outside this repo):
  git clone https://github.com/yuxumin/PoinTr
  # install per their README (pointnet2_ops, KNN_CUDA, etc.); download the
  # AdaPoinTr checkpoint (e.g. trained on PCN or ShapeNet-55).
  set POINTR_ROOT=C:\path\to\PoinTr
  set POINTR_CKPT=C:\path\to\adapointr.pth

Then:
  python scripts/run_baseline_adapointr.py
  python scripts/eval_completion_dir.py \
     --completed_dir data/completion_test_samples/test_set/completed_adapointr \
     --suffix _completed_adapointr.ply --method AdaPoinTr \
     --out_csv data/completion_test_samples/baseline_results.csv
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, 'data',
                                'completion_test_samples', 'test_set')
TEMPLATE_PLY = os.path.join(PROJECT_ROOT, 'data', 'template.ply')


def load_adapointr(pointr_root, ckpt):
    """Import the external AdaPoinTr model. Returns a callable infer(pc)->pc."""
    if not pointr_root or not os.path.isdir(pointr_root):
        sys.exit('Set POINTR_ROOT to your cloned PoinTr repo (see header).')
    if not ckpt or not os.path.isfile(ckpt):
        sys.exit('Set POINTR_CKPT to the AdaPoinTr checkpoint (see header).')
    sys.path.insert(0, pointr_root)
    try:
        import torch
        from tools import builder
        from utils.config import cfg_from_yaml_file
    except Exception as ex:
        sys.exit(f'Could not import PoinTr ({ex}). Install its deps per their README.')

    cfg_path = os.path.join(pointr_root, 'cfgs', 'PCN_models', 'AdaPoinTr.yaml')
    config = cfg_from_yaml_file(cfg_path)
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt)
    model = model.cuda().eval()

    def infer(pc):  # pc: (M,3) numpy, normalized to unit sphere by caller
        with torch.no_grad():
            x = torch.from_numpy(pc).float().cuda().unsqueeze(0)
            out = model(x)
            dense = out[-1] if isinstance(out, (list, tuple)) else out
            return dense.squeeze(0).cpu().numpy()
    return infer


def normalize_unit_sphere(pc):
    c = pc.mean(0)
    p = pc - c
    s = np.linalg.norm(p, axis=1).max() + 1e-9
    return p / s, c, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', default=DEFAULT_TEST_DIR)
    ap.add_argument('--template', default=TEMPLATE_PLY)
    ap.add_argument('--pointr_root', default=os.environ.get('POINTR_ROOT'))
    ap.add_argument('--ckpt', default=os.environ.get('POINTR_CKPT'))
    args = ap.parse_args()

    infer = load_adapointr(args.pointr_root, args.ckpt)

    tmpl = trimesh.load(args.template, process=False)
    faces = np.asarray(tmpl.faces)
    n = len(tmpl.vertices)

    out_dir = os.path.join(args.test_dir, 'completed_adapointr')
    os.makedirs(out_dir, exist_ok=True)

    for pf in sorted(glob(os.path.join(args.test_dir, '*_partial_*.ply'))):
        basename = os.path.basename(pf).split('_partial_')[0]
        mask_file = os.path.join(args.test_dir, f'{basename}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        partial = np.asarray(trimesh.load(pf, process=False).vertices, dtype=np.float64)
        mask = np.load(mask_file).astype(bool)
        known = ~mask

        known_pts = partial[known]
        pc_in, c, s = normalize_unit_sphere(known_pts)
        pred = infer(pc_in.astype(np.float32))
        pred = pred * s + c                       # back to data scale

        # remap: missing template vertices <- nearest predicted point
        tree = cKDTree(pred)
        comp = partial.copy()
        _, idx = tree.query(_template_guess(partial, mask), k=1)
        comp[mask] = pred[idx]
        comp[known] = partial[known]              # keep observed exactly

        out_path = os.path.join(out_dir, f'{basename}_completed_adapointr.ply')
        trimesh.Trimesh(vertices=comp, faces=faces, process=False).export(out_path)
        print(f'  {basename}: -> {out_path}')

    print(f'\nDone. See {out_dir}. Score with eval_completion_dir.py.')


def _template_guess(partial, mask):
    """Query positions for missing vertices: use the template-mean position as the
    lookup point (we don't know the true position). Falls back to centroid."""
    tmpl = trimesh.load(TEMPLATE_PLY, process=False)
    t = np.asarray(tmpl.vertices, dtype=np.float64)
    return t[mask]


if __name__ == '__main__':
    main()
