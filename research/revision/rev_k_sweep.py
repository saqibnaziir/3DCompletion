"""
Paper-revision measurement 3: uncertainty-vs-error agreement as a function of K.

Same test set, same estimator, same metric code as the existing K=20 run:
    u_i = || std_k v_i^(k) ||_2          (numpy std, ddof=0 -- unchanged)
    e_i = || mean_k v_i^(k) - gt_i ||_2
    metrics via scripts.eval_uncertainty_calibration.metrics_from (Spearman,
    Pearson, AUSE, AURG), pooled over all missing vertices of all meshes.

SEED PROTOCOL. eval_uncertainty_calibration.py seeds once
(torch.manual_seed(0); np.random.seed(0)) and then streams draws mesh by mesh.
This script does the same, but draws a POOL of --pool samples per mesh in one
call, then slices the pool into disjoint blocks:

    K=20 : pool[0:20]                            <- the reference configuration
    K=10 : pool[0:10], pool[10:20], pool[20:30]  <- 3 disjoint replicates
    K=5  : pool[0:5],  pool[5:10], pool[10:15]   <- 3 disjoint replicates

Replicates at a given K are disjoint, so they are independent draws and their
spread is a real estimate of subset luck. The K=20 block is the pool prefix, so
the K=5/K=10 numbers are compared against it on the SAME underlying samples --
the difference is the effect of K, not of a different sampling run.

Read-only: loads checkpoints, writes only CSV/JSON under --out_dir.

Run (conda env: 3dmesh)
  python research/revision/rev_k_sweep.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import csv
import json
import time
from glob import glob

import numpy as np
import torch
from scipy.stats import spearmanr
from trimesh import load_mesh

from config.config import read_config, resolve_config_file
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)
from research.eval_uncertainty_calibration import metrics_from

# K -> list of (start, stop) slices into the per-mesh pool
BLOCKS = {
    5:  [(0, 5), (5, 10), (10, 15)],
    10: [(0, 10), (10, 20), (20, 30)],
    20: [(0, 20)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--param', default='x0')
    ap.add_argument('--tag', default='depthw_nrm_scratch')
    ap.add_argument('--test_dir',
                    default=os.path.join('data', 'completion_test_samples', 'test_set'))
    ap.add_argument('--pool', type=int, default=30, help='draws per mesh')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=os.path.join('data', '_revision'))
    args = ap.parse_args()

    cfg_path = resolve_config_file(args.data, args.config_file)
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    dcfg['predict'] = args.param
    arm = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    denoiser = build_denoiser(config, dcfg, device, use_ema=True)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = get_template_info(config)
    N = len(template_mesh.vertices)
    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma = scale['sigma'].to(device) if isinstance(scale, dict) else scale.to(device)
    clip_value = scale.get('clip_value', None) if isinstance(scale, dict) else None

    gt_dir = os.path.join(args.test_dir, 'gt')
    mask_dir = os.path.join(args.test_dir, 'partial')
    cases = []
    for gt_file in sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply'))):
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mf = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mf):
            continue
        mask = np.load(mf)
        gt = np.asarray(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N or len(mask) != N or not mask.any():
            continue
        partial = gt.copy(); partial[mask] = 0.0
        cases.append((base, partial, mask, gt))

    print(f"arm    : {arm}")
    print(f"device : {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"meshes : {len(cases)} | pool={args.pool} draws/mesh | {steps} steps\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- one sampling pass; every K is scored off the same pool ----
    pools = {}
    for base, partial, mask, gt in cases:
        t0 = time.perf_counter()
        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value, num_steps=steps,
                           num_samples=args.pool)
        pools[base] = np.stack([s.astype(np.float64) for s in samples], axis=0)
        print(f"  {base}: {args.pool} draws [{time.perf_counter()-t0:.1f}s]")

    # ---- score every (K, replicate) ----
    rows, per_mesh_rows = [], []
    for K in sorted(BLOCKS):
        for rep, (lo, hi) in enumerate(BLOCKS[K]):
            all_u, all_e = [], []
            mesh_u, mesh_e = [], []
            for base, partial, mask, gt in cases:
                stack = pools[base][lo:hi]                    # (K,N,3)
                mu = stack.mean(axis=0)
                u = np.linalg.norm(stack.std(axis=0), axis=1)
                e = np.linalg.norm(mu - gt.astype(np.float64), axis=1)
                um, em = u[mask], e[mask]
                all_u.append(um); all_e.append(em)
                mesh_u.append(um.mean()); mesh_e.append(em.mean())
                pm, _ = metrics_from(um, em)
                per_mesh_rows.append({'K': K, 'rep': rep, 'mesh': base,
                                      'spearman_rho': pm['spearman_rho'],
                                      'pearson_r': pm['pearson_r'],
                                      'ause': pm['ause'], 'aurg': pm['aurg'],
                                      'mean_u_mm': 1000.0 * um.mean(),
                                      'mean_e_mm': 1000.0 * em.mean()})

            U = np.concatenate(all_u); E = np.concatenate(all_e)
            agg, _ = metrics_from(U, E)
            btw = float(spearmanr(np.array(mesh_u), np.array(mesh_e))[0])
            rows.append({
                'K': K, 'rep': rep, 'block': f'[{lo}:{hi}]',
                'n_vertices': int(len(E)), 'n_meshes': len(cases),
                'spearman_rho': agg['spearman_rho'], 'pearson_r': agg['pearson_r'],
                'ause': agg['ause'], 'aurg': agg['aurg'],
                'mean_uncertainty_mm': 1000.0 * agg['mean_uncertainty'],
                'mean_error_mm': 1000.0 * agg['mean_error'],
                'between_mesh_spearman': btw,
                'per_mesh_spearman_mean': float(np.mean(
                    [r['spearman_rho'] for r in per_mesh_rows
                     if r['K'] == K and r['rep'] == rep])),
            })
            print(f"K={K:>2} rep{rep} {rows[-1]['block']:>9}: "
                  f"rho={agg['spearman_rho']:.4f} r={agg['pearson_r']:.4f} "
                  f"AUSE={agg['ause']:.4f} AURG={agg['aurg']:.4f} "
                  f"u={1000*agg['mean_uncertainty']:.4f}mm "
                  f"e={1000*agg['mean_error']:.4f}mm btw={btw:+.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'k_sweep_reps.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, 'k_sweep_per_mesh.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_mesh_rows[0].keys()))
        w.writeheader(); w.writerows(per_mesh_rows)

    # ---- mean +/- sd over replicates ----
    KEYS = ['spearman_rho', 'pearson_r', 'ause', 'aurg',
            'mean_uncertainty_mm', 'mean_error_mm', 'between_mesh_spearman']
    summary = {}
    W = 108
    print("\n" + "=" * W)
    print(f"UNCERTAINTY-vs-ERROR AGREEMENT vs K   (arm={arm}, {len(cases)} meshes, "
          f"pooled over {rows[0]['n_vertices']} missing vertices)")
    print("-" * W)
    print(f"{'K':>3}{'reps':>6}{'Spearman':>19}{'Pearson':>19}{'AUSE':>17}"
          f"{'AURG':>17}{'btw-mesh rho':>16}")
    for K in sorted(BLOCKS):
        sub = [r for r in rows if r['K'] == K]
        s = {k: (float(np.mean([r[k] for r in sub])),
                 float(np.std([r[k] for r in sub], ddof=1)) if len(sub) > 1 else 0.0)
             for k in KEYS}
        summary[K] = {'n_reps': len(sub),
                      **{k: {'mean': v[0], 'sd': v[1]} for k, v in s.items()}}
        f2 = lambda k, p=4: (f"{s[k][0]:.{p}f}+/-{s[k][1]:.{p}f}"
                             if len(sub) > 1 else f"{s[k][0]:.{p}f}")
        print(f"{K:>3}{len(sub):>6}{f2('spearman_rho'):>19}{f2('pearson_r'):>19}"
              f"{f2('ause'):>17}{f2('aurg'):>17}{f2('between_mesh_spearman',3):>16}")
    print("-" * W)
    print(f"{'K':>3}{'reps':>6}{'mean uncertainty (mm)':>28}{'mean error (mm)':>26}")
    for K in sorted(BLOCKS):
        sub = [r for r in rows if r['K'] == K]
        mu = np.mean([r['mean_uncertainty_mm'] for r in sub])
        su = np.std([r['mean_uncertainty_mm'] for r in sub], ddof=1) if len(sub) > 1 else 0.0
        me = np.mean([r['mean_error_mm'] for r in sub])
        se = np.std([r['mean_error_mm'] for r in sub], ddof=1) if len(sub) > 1 else 0.0
        a = f"{mu:.4f}+/-{su:.4f}" if len(sub) > 1 else f"{mu:.4f}"
        b = f"{me:.4f}+/-{se:.4f}" if len(sub) > 1 else f"{me:.4f}"
        print(f"{K:>3}{len(sub):>6}{a:>28}{b:>26}")
    print("=" * W)

    with open(os.path.join(args.out_dir, 'k_sweep_summary.json'), 'w') as f:
        json.dump({'arm': arm, 'n_meshes': len(cases), 'pool': args.pool,
                   'steps': steps, 'seed': args.seed, 'blocks':
                   {str(k): v for k, v in BLOCKS.items()},
                   'summary': {str(k): v for k, v in summary.items()}}, f, indent=1)
    print(f"\nwrote {args.out_dir}/k_sweep_*.csv + .json")


if __name__ == '__main__':
    main()
