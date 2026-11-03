"""
Dump K completions for a few chosen test meshes to a .npz, so the figure script
can be iterated on without re-running the sampler every time.

  python scripts/dump_hero_samples.py --config_file config/config_diffusion.cfg \
         --param x0 --tag depthw_nrm_scratch --K 20 --meshes test_004 test_010
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
from trimesh import load_mesh

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (
    parse_diffusion_section, build_denoiser, get_template_info, complete,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--param', default='x0', choices=['eps', 'x0', 'v'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--meshes', nargs='+', required=True)
    ap.add_argument('--test_dir', default='data/completion_test_samples/test_set')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  arm={arm}  K={args.K}  steps={num_steps}')

    denoiser = build_denoiser(config, dcfg, device, use_ema=True)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces)
    N_t = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device); clip_value = scale.get('clip_value', None)
    else:
        sigma = scale.to(device); clip_value = None

    gt_dir = os.path.join(args.test_dir, 'gt')
    mask_dir = os.path.join(args.test_dir, 'partial')

    out = {'faces': faces, 'meshes': np.array(args.meshes), 'K': args.K}
    for base in args.meshes:
        gt = np.asarray(load_mesh(os.path.join(gt_dir, f'{base}_gt_complete.ply'),
                                  process=False).vertices, dtype=np.float64)
        mask = np.load(os.path.join(mask_dir, f'{base}_mask.npy'))
        assert len(gt) == N_t and len(mask) == N_t, f'{base}: size mismatch'

        partial = gt.copy().astype(np.float32); partial[mask] = 0.0
        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value, num_steps=num_steps,
                           num_samples=args.K)
        stack = np.stack([s.astype(np.float64) for s in samples], axis=0)  # (K,N,3)

        out[f'{base}_stack'] = stack
        out[f'{base}_gt'] = gt
        out[f'{base}_mask'] = mask
        u = np.linalg.norm(stack.std(axis=0), axis=1)
        e = np.linalg.norm(stack.mean(axis=0) - gt, axis=1)
        print(f'  {base}: miss={mask.sum():5d}  '
              f'unc={1000*u[mask].mean():.3f}mm  err={1000*e[mask].mean():.3f}mm')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
