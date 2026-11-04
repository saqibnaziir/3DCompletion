"""
Export example completions from the official test benchmark for visual checks.

Picks N samples spread evenly over the three mask types and, within each type,
over the full hole-size range (evenly spaced quantiles of missing %). For each:
    <name>_gt.ply          ground truth
    <name>_partial.ply     input: missing vertices + touching faces removed
    <name>_completed.ply   model output (known vertices identical to GT)
    <name>_errmap.ply      completed mesh coloured by per-vertex error (0 -> 5 mm)
    <name>_mask.npy        True = missing
plus examples.csv with the errors.

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/export_examples.py --param x0 --tag depthw_nrm --n 20
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
from trimesh import Trimesh, load_mesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from utils.funcs import save_ply_explicit
from utils.completion_dataset import remove_masked_region
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)

STRATEGIES = ('random_scatter', 'bfs_patch', 'plane_cut')
ERR_MAX_MM = 5.0


def error_colors(err_mm):
    """Grey -> yellow -> red ramp, saturating at ERR_MAX_MM."""
    t = np.clip(err_mm / ERR_MAX_MM, 0.0, 1.0)[:, None]
    lo, mid, hi = np.array([200, 200, 200]), np.array([255, 210, 0]), np.array([220, 20, 20])
    c = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    return np.concatenate([c, np.full((len(c), 1), 255)], 1).astype(np.uint8)


def pick(index, n):
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    chosen = []
    for strat, k in zip(STRATEGIES, per):
        rows = sorted([r for r in index if r['strategy'] == strat],
                      key=lambda r: float(r['actual_ratio']))
        idx = np.linspace(0, len(rows) - 1, k).round().astype(int)
        chosen += [rows[i] for i in idx]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file',
                    default=os.path.join(_ROOT, 'config', 'config_diffusion_revision.cfg'))
    ap.add_argument('--param', required=True, choices=['eps', 'x0', 'v'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--bench_dir', default=os.path.join(_ROOT, 'revision', 'benchmark'))
    ap.add_argument('--out_dir', default=None,
                    help='default: revision/results/official_test/<arm>/examples')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm = f'predict_{args.param}' + (f'_{args.tag}' if args.tag else '')
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    out_dir = args.out_dir or os.path.join(_ROOT, 'revision', 'results', 'official_test',
                                           arm, 'examples')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    denoiser = build_denoiser(config, dcfg, device, use_ema=True)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'], schedule=dcfg['schedule'],
                                  predict=dcfg['predict']).to(device)
    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces)
    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma, clip_value = scale['sigma'].to(device), scale.get('clip_value')

    masks = np.load(os.path.join(args.bench_dir, 'official_test_masks.npz'))['masks']
    with open(os.path.join(args.bench_dir, 'official_test_index.csv')) as fh:
        index = list(csv.DictReader(fh))

    rows = []
    for i, rec in enumerate(pick(index, args.n)):
        mask = masks[int(rec['row'])]
        gt = np.asarray(load_mesh(os.path.join(_ROOT, rec['gt_path']), process=False).vertices,
                        dtype=np.float32)
        partial = gt.copy()
        partial[mask] = 0.0
        comp = complete(denoiser, diffusion, partial, mask, config, device, sigma,
                        clip_value=clip_value, num_steps=dcfg['sampling_steps'],
                        num_samples=1)[0]
        err = np.linalg.norm(comp - gt, axis=1) * 1000.0

        pct = 100.0 * mask.mean()
        name = (f"{i:02d}_{rec['strategy']}_{pct:04.1f}pct_{rec['subject']}_"
                f"{rec['expression']}.{rec['frame']}")
        save_ply_explicit(Trimesh(gt, faces, process=False),
                          os.path.join(out_dir, name + '_gt.ply'))
        pv, pf = remove_masked_region(gt, faces, mask)
        save_ply_explicit(Trimesh(pv, pf, process=False),
                          os.path.join(out_dir, name + '_partial.ply'))
        save_ply_explicit(Trimesh(comp, faces, process=False),
                          os.path.join(out_dir, name + '_completed.ply'))
        Trimesh(comp, faces, vertex_colors=error_colors(err), process=False).export(
            os.path.join(out_dir, name + '_errmap.ply'))
        np.save(os.path.join(out_dir, name + '_mask.npy'), mask)

        rows.append(dict(name=name, strategy=rec['strategy'], missing_pct=round(pct, 1),
                         missing_mm=round(float(err[mask].mean()), 3),
                         max_mm=round(float(err.max()), 3),
                         known_max_mm=float(err[~mask].max())))
        print(f"  {name}: missing {rows[-1]['missing_mm']:.3f} mm "
              f"(max {rows[-1]['max_mm']:.2f}, known {rows[-1]['known_max_mm']:.1e})")

    with open(os.path.join(out_dir, 'examples.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('saved to', out_dir)


if __name__ == '__main__':
    main()
