"""
Paper-revision measurement 2: wall-clock seconds per mesh, batch size 1.

Read-only: loads existing checkpoints, trains nothing, writes only a CSV/JSON of
timings. Uses scripts.infer_diffusion.complete() unchanged so the timed code path
is exactly the inference path the paper's numbers come from.

Two things are hoisted OUT of the timed region so the number is compute-only:
  * every mesh + mask is read from disk before timing starts;
  * load_norm() (which re-reads norm_completion.pt on every call) is memoised.
GPU work is bracketed with torch.cuda.synchronize().

Run (conda env: 3dmesh)
  python scripts/_revision/rev_runtime.py --K 20 --meshes 20
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
from trimesh import load_mesh

from config.config import read_config, resolve_config_file
from utils.diffusion import GaussianDiffusion
from scripts import infer_diffusion as ID


def memoise_load_norm():
    """Stop complete() re-reading norm_completion.pt on every single call."""
    cache = {}
    original = ID.load_norm

    def cached(config):
        key = config['dataset_dir']
        if key not in cache:
            cache[key] = original(config)
        return cache[key]

    ID.load_norm = cached
    return original


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def batched_sample(denoiser, diffusion, cond1, N, K, device, clip_value, steps):
    """K samples in ONE forward batch (B=K), for contrast with the sequential loop.

    Not what scripts/infer_diffusion.py does today -- reported separately so the
    'are samples batched?' question has a measured answer on both sides.
    """
    cond = cond1.repeat(K, 1)
    known_mask = cond[:, 3] > 0.5
    known_values = cond[:, :3].contiguous()
    return diffusion.ddim_sample(
        denoiser, cond, B=K, nodes_per_graph=N,
        known_mask=known_mask, known_values=known_values,
        num_steps=steps, device=device, progress=False, clip_value=clip_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--param', default='x0')
    ap.add_argument('--tag', default='depthw_nrm_scratch')
    ap.add_argument('--test_dir',
                    default=os.path.join('data', 'completion_test_samples', 'test_set'))
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--meshes', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=os.path.join('data', '_revision'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = resolve_config_file(args.data, args.config_file)
    config = read_config(cfg_path)
    dcfg = ID.parse_diffusion_section(cfg_path)
    dcfg['predict'] = args.param
    arm = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    steps = args.steps if args.steps is not None else dcfg['sampling_steps']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f"arm    : {arm}")
    print(f"device : {device}  ({gpu_name})")
    print(f"torch  : {torch.__version__}")

    memoise_load_norm()

    denoiser = ID.build_denoiser(config, dcfg, device, use_ema=True)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = ID.get_template_info(config)
    N = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma = scale['sigma'].to(device) if isinstance(scale, dict) else scale.to(device)
    clip_value = scale.get('clip_value', None) if isinstance(scale, dict) else None

    # ---- preload EVERYTHING from disk before any timing ----
    gt_dir = os.path.join(args.test_dir, 'gt')
    mask_dir = os.path.join(args.test_dir, 'partial')
    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))[:args.meshes]
    cases = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mf = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mf):
            continue
        mask = np.load(mf)
        gt = np.asarray(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N or len(mask) != N:
            continue
        partial = gt.copy()
        partial[mask] = 0.0
        cases.append((base, partial, mask))
    print(f"meshes : {len(cases)} preloaded | N={N} verts | {steps} DDIM steps\n")

    # ---- warm-up (cuDNN autotune, edge-index cache, lazy CUDA init) ----
    for _ in range(args.warmup):
        ID.complete(denoiser, diffusion, cases[0][1], cases[0][2], config, device,
                    sigma, clip_value=clip_value, num_steps=steps, num_samples=1)
    sync(device)
    print("warm-up done\n")

    rows = []
    print(f"{'sample':<10} {'K=1 (s)':>10} {'K=%d seq (s)' % args.K:>14} "
          f"{'K=%d batched (s)' % args.K:>17}")
    print("-" * 55)
    for base, partial, mask in cases:
        sync(device)
        t0 = time.perf_counter()
        ID.complete(denoiser, diffusion, partial, mask, config, device, sigma,
                    clip_value=clip_value, num_steps=steps, num_samples=1)
        sync(device)
        t_k1 = time.perf_counter() - t0

        sync(device)
        t0 = time.perf_counter()
        ID.complete(denoiser, diffusion, partial, mask, config, device, sigma,
                    clip_value=clip_value, num_steps=steps, num_samples=args.K)
        sync(device)
        t_kseq = time.perf_counter() - t0

        # batched contrast: build cond exactly as complete() does, then B=K
        mean, std, mean_only = ID.load_norm(config)
        vn = partial.astype(np.float32).copy() - mean.numpy()
        if not mean_only and std is not None:
            vn = vn / std.numpy()
        vn[mask] = 0.0
        cond_xyz = vn / sigma.cpu().numpy().astype(np.float32)
        cond_xyz[mask] = 0.0
        mch = np.ones((N, 1), np.float32)
        mch[mask] = 0.0
        cond1 = torch.tensor(np.concatenate([cond_xyz, mch], 1), device=device)
        try:
            sync(device)
            t0 = time.perf_counter()
            batched_sample(denoiser, diffusion, cond1, N, args.K, device,
                           clip_value, steps)
            sync(device)
            t_kbat = time.perf_counter() - t0
        except RuntimeError as exc:            # OOM on an 8 GB laptop card
            t_kbat = float('nan')
            print(f"  (batched K={args.K} failed: {type(exc).__name__})")

        rows.append({'sample': base, 't_k1_s': t_k1, 't_kseq_s': t_kseq,
                     't_kbatched_s': t_kbat, 'K': args.K, 'steps': steps})
        print(f"{base:<10} {t_k1:>10.3f} {t_kseq:>14.3f} {t_kbat:>17.3f}")

    a = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    k1, kseq, kbat = a('t_k1_s'), a('t_kseq_s'), a('t_kbatched_s')

    print("-" * 55)
    print(f"{'MEAN':<10} {k1.mean():>10.3f} {kseq.mean():>14.3f} "
          f"{np.nanmean(kbat):>17.3f}")
    print(f"{'STD':<10} {k1.std(ddof=1):>10.3f} {kseq.std(ddof=1):>14.3f} "
          f"{np.nanstd(kbat, ddof=1):>17.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'runtime_ours.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {
        'gpu': gpu_name, 'torch': torch.__version__, 'arm': arm,
        'n_meshes': len(rows), 'n_vertices': N, 'ddim_steps': steps, 'K': args.K,
        'batch_size': 1, 'warmup_runs': args.warmup,
        'k1_mean_s': float(k1.mean()), 'k1_std_s': float(k1.std(ddof=1)),
        'kseq_mean_s': float(kseq.mean()), 'kseq_std_s': float(kseq.std(ddof=1)),
        'kbatched_mean_s': float(np.nanmean(kbat)),
        'samples_batched_in_repo_code': False,
    }
    with open(os.path.join(args.out_dir, 'runtime_ours.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {args.out_dir}/runtime_ours.csv + .json")


if __name__ == '__main__':
    main()
