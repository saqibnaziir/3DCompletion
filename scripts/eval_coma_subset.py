"""
Per-actor completion error on data/COMA_test_subset, for comparison with the
Anisotropic SpiralNet (Sensors 2022) Table 3 protocol (half the vertices removed).

Reports mean per-vertex Euclidean error in mm, both over ALL vertices
(comparable to their table) and over the missing region only.

  python scripts/eval_coma_subset.py --config_file config/config_diffusion.cfg \
         --param x0 --tag depthw_nrm_scratch --K 1
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import numpy as np
import torch
import trimesh
from collections import defaultdict
from scipy.spatial import cKDTree

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)

N = 5023


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--param', default='x0', choices=['eps', 'x0', 'v'])
    ap.add_argument('--tag', default='depthw_nrm_scratch')
    ap.add_argument('--K', type=int, default=1, help='samples per mesh (mean-of-K)')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--subset_dir', default='data/COMA_test_subset')
    ap.add_argument('--out_csv', default='data/COMA_test_subset/our_results.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    files = sorted(glob.glob(os.path.join(args.subset_dir, 'gt', '*', '*.ply')))
    print(f'{len(files)} meshes')

    # ---- fixed mask, recovered once from any partial ----
    g0 = files[0]
    p0 = g0.replace(os.sep + 'gt' + os.sep, os.sep + 'partail' + os.sep) \
           .replace('.ply', '_subset.ply')
    gv = np.asarray(trimesh.load(g0, process=False).vertices)
    pv = np.asarray(trimesh.load(p0, process=False).vertices)
    _, idx = cKDTree(gv).query(pv)
    keep = np.zeros(N, bool); keep[idx] = True
    mask = ~keep
    print(f'mask: {mask.sum()} missing ({100*mask.mean():.1f}%)')

    # ---- flag meshes that are in our training split ----
    norm = torch.load('data/processed_completion/norm.pt', weights_only=False)
    mean = np.asarray(norm['mean'])
    d, _ = torch.load('data/processed_completion/training.pt', weights_only=False)
    y = d.y if getattr(d, 'y', None) is not None else d.x
    n = y.shape[0] // N
    vi = np.linspace(0, N - 1, 200).astype(int)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TR = torch.tensor((y[:n*N].reshape(n, N, 3).numpy() + mean)[:, vi].reshape(n, -1),
                      dtype=torch.float32, device=dev)
    GT = np.stack([np.asarray(trimesh.load(f, process=False).vertices) for f in files])
    Q = torch.tensor(GT[:, vi].reshape(len(files), -1), dtype=torch.float32, device=dev)
    in_train = ((torch.cdist(Q, TR).min(1).values / np.sqrt(200) * 1000)
                .cpu().numpy() < 0.01)
    del TR, Q
    print(f'in our training split: {in_train.sum()}/{len(files)}')

    # ---- model ----
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm = f'predict_{args.param}' + (f'_{args.tag}' if args.tag else '')
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    den = build_denoiser(config, dcfg, dev, use_ema=True)
    diff = GaussianDiffusion(timesteps=dcfg['timesteps'], schedule=dcfg['schedule'],
                             predict=args.param).to(dev)
    sc = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                    map_location=dev, weights_only=False)
    sigma = sc['sigma'].to(dev) if isinstance(sc, dict) else sc.to(dev)
    clip = sc.get('clip_value') if isinstance(sc, dict) else None
    print(f'arm={arm}  K={args.K}  steps={steps}\n')

    rows = []
    for i, f in enumerate(files):
        gt = GT[i].astype(np.float32)
        partial = gt.copy(); partial[mask] = 0.0
        s = complete(den, diff, partial, mask, config, dev, sigma, clip_value=clip,
                     num_steps=steps, num_samples=args.K)
        out = np.stack(s, 0).mean(0) if args.K > 1 else s[0]
        e = np.linalg.norm(out - gt, axis=1) * 1000.0
        rows.append({'actor': os.path.basename(os.path.dirname(f)),
                     'mesh': os.path.basename(f), 'in_train': bool(in_train[i]),
                     'full': float(e.mean()), 'hole': float(e[mask].mean())})
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)}')

    # ---- per-actor table ----
    by = defaultdict(list)
    for r in rows:
        by[r['actor']].append(r)
    actors = sorted(by)

    def agg(sel, key):
        v = [r[key] for r in sel]
        return float(np.mean(v)) if v else float('nan')

    print(f'\n{"actor":<20}{"n":>5}{"full(all)":>11}{"hole(all)":>11}'
          f'{"n_clean":>9}{"full(cln)":>11}{"hole(cln)":>11}')
    for a in actors:
        s = by[a]; c = [r for r in s if not r['in_train']]
        print(f'{a:<20}{len(s):>5}{agg(s,"full"):>11.3f}{agg(s,"hole"):>11.3f}'
              f'{len(c):>9}{agg(c,"full"):>11.3f}{agg(c,"hole"):>11.3f}')
    clean = [r for r in rows if not r['in_train']]
    print(f'{"MEAN":<20}{len(rows):>5}{agg(rows,"full"):>11.3f}{agg(rows,"hole"):>11.3f}'
          f'{len(clean):>9}{agg(clean,"full"):>11.3f}{agg(clean,"hole"):>11.3f}')

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w') as fh:
        fh.write('actor,mesh,in_train,full_mm,hole_mm\n')
        for r in rows:
            fh.write(f'{r["actor"]},{r["mesh"]},{int(r["in_train"])},'
                     f'{r["full"]:.5f},{r["hole"]:.5f}\n')
    print(f'\nwritten: {args.out_csv}')


if __name__ == '__main__':
    main()
