"""
Inference-time ablations for the diffusion completer (fills paper Table 3 / §4.5).

Holds the trained model fixed and toggles ONE inference knob at a time, scoring
the SAME per-vertex metric as the main table on the same test set:

  (b) x0-clamp        : clip_value = saved  vs  None (off)
  (c) DDIM steps      : {10, 20, 50, 100}
  (e) data-consistency: per-step known-region projection ON vs OFF
      (OFF = run the sampler with an all-False known_mask so the RePaint
       overwrite/re-noise becomes a no-op; the model still SEES the observation
       through the conditioning, it is just not hard-projected each step. The
       final output still pastes the known region verbatim, as always.)

The 'full' config (clip=saved, steps=50, DC=on) should reproduce the headline
number (~0.00207), a built-in sanity check.

Ablations (a) standardization and (d) diffusion-vs-regressor are TRAINING-time
and handled elsewhere (standardization-off diverges during training; the
deterministic regressor is scripts/baseline_regressor.py).

All configs share the same random seed and mesh order, so comparisons are paired
(same initial noise per mesh across configs) -> low-variance ablation.

Run (GPU box with the trained diffusion checkpoint):
  python scripts/ablation_diffusion.py --config_file config/config_diffusion.cfg \
         --test_dir data/completion_test_samples/test_set
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
from glob import glob
from trimesh import load_mesh

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (
    parse_diffusion_section, build_denoiser, get_template_info, load_norm,
)


@torch.no_grad()
def complete_one(denoiser, diffusion, verts_np, mask, config, device, sigma,
                 clip_value, num_steps, dc_on):
    """One completion (N,3) in original coords; known region exact via final blend."""
    mean, std, mean_only = load_norm(config)
    mean_np = mean.numpy()
    std_np = std.numpy() if std is not None else None
    sigma_np = sigma.cpu().numpy().astype(np.float32)
    N = len(verts_np)

    vn = verts_np.astype(np.float32).copy() - mean_np
    if not mean_only and std_np is not None:
        vn = vn / std_np
    vn[mask] = 0.0
    cond_xyz = vn / sigma_np
    cond_xyz[mask] = 0.0
    mch = np.ones((N, 1), np.float32); mch[mask] = 0.0
    cond = torch.tensor(np.concatenate([cond_xyz, mch], 1), device=device)

    real_known = cond[:, 3] > 0.5
    known_values = cond[:, :3].contiguous()
    known_mask = real_known if dc_on else torch.zeros_like(real_known)

    x0 = diffusion.ddim_sample(denoiser, cond, B=1, nodes_per_graph=N,
                               known_mask=known_mask, known_values=known_values,
                               num_steps=num_steps, device=device,
                               clip_value=clip_value)
    out = x0.cpu().numpy() * sigma_np
    if not mean_only and std_np is not None:
        out = out * std_np
    out = out + mean_np
    final = verts_np.copy(); final[mask] = out[mask]
    return final


def score_config(denoiser, diffusion, config, device, sigma, gt_files, N_t,
                 clip_value, num_steps, dc_on, seed, mask_dir):
    torch.manual_seed(seed); np.random.seed(seed)         # paired across configs
    miss, over = [], []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        mask = np.load(mask_file).astype(bool)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t or len(mask) != N_t or not mask.any():
            continue
        partial = gt.copy(); partial[mask] = 0.0
        comp = complete_one(denoiser, diffusion, partial, mask, config, device,
                            sigma, clip_value, num_steps, dc_on)
        e = np.linalg.norm(comp - gt, axis=1)
        miss.append(float(e[mask].mean())); over.append(float(e.mean()))
    return (float(np.mean(miss)), float(np.std(miss)),
            float(np.mean(over)), float(np.std(over)), len(miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--param', default='eps', choices=['eps', 'x0', 'v'],
                    help="prediction target of the arm being scored. MUST match how "
                         "the arm was trained -- decoding an x0-trained model as eps "
                         "silently produces numbers for a model that does not exist.")
    ap.add_argument('--tag', default=None,
                    help="checkpoint subfolder suffix -> predict_<param>[_<tag>]/")
    ap.add_argument('--test_dir', default=None)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  ckpt={arm}')

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = get_template_info(config)
    N_t = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device); saved_clip = scale.get('clip_value', None)
    else:
        sigma = scale.to(device); saved_clip = None
    print(f'saved clip_value={saved_clip}')

    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples', 'test_set')
    # Split layout writes <test_dir>/gt and <test_dir>/partial; flat layout keeps
    # both beside each other. Globbing test_dir blindly finds zero meshes on the
    # split layout and reports an empty table rather than failing.
    gt_dir = (os.path.join(test_dir, 'gt')
              if os.path.isdir(os.path.join(test_dir, 'gt')) else test_dir)
    mask_dir = (os.path.join(test_dir, 'partial')
                if os.path.isdir(os.path.join(test_dir, 'partial')) else test_dir)
    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if not gt_files:
        raise SystemExit(f'no *_gt_complete.ply under {gt_dir}')
    print(f'{len(gt_files)} test meshes  (gt={gt_dir}, masks={mask_dir})\n')

    # (label, clip_value, steps, dc_on)
    # NOTE: the per-step-projection arm lives in scripts/ablation_consistency.py,
    # which scores it against classical conditional diffusion and final-paste-only
    # on known-region and seam metrics too. The row below toggles only the
    # per-step projection and STILL pastes the observation at the end, so it is
    # "hard consistency at the end only" -- not unconditioned diffusion.
    configs = [
        ('full (clip,DC,50)',      saved_clip, 50, True),
        ('no x0-clamp',            None,       50, True),
        ('DC at end only (no per-step proj)', saved_clip, 50, False),
        ('DDIM steps=10',          saved_clip, 10, True),
        ('DDIM steps=20',          saved_clip, 20, True),
        ('DDIM steps=100',         saved_clip, 100, True),
    ]

    rows = []
    print(f'{"config":24s} {"missing":>18s} {"overall":>18s}   n')
    print('-' * 70)
    for label, clip_v, steps, dc in configs:
        mm, ms, om, os_, n = score_config(denoiser, diffusion, config, device,
                                          sigma, gt_files, N_t, clip_v, steps, dc,
                                          args.seed, mask_dir)
        rows.append((label, clip_v, steps, dc, mm, ms, om, os_, n))
        print(f'{label:24s} {mm:.5f}+/-{ms:.5f} {om:.5f}+/-{os_:.5f}  {n}')

    out_csv = args.out_csv or os.path.join(test_dir, 'ablation_results.csv')
    with open(out_csv, 'w') as f:
        f.write('config,clip_value,steps,data_consistency,'
                'missing_mean,missing_std,overall_mean,overall_std,n\n')
        for (label, clip_v, steps, dc, mm, ms, om, os_, n) in rows:
            f.write(f'{label},{clip_v},{steps},{int(dc)},'
                    f'{mm:.6f},{ms:.6f},{om:.6f},{os_:.6f},{n}\n')
    print(f'\nwrote {out_csv}')

    # LaTeX-ready snippet (missing-region, mm)
    print('\n% --- paste-ready numbers (missing-region error; mm = x1000) ---')
    for (label, clip_v, steps, dc, mm, ms, om, os_, n) in rows:
        print(f'%   {label:24s} {mm:.5f}  ({mm*1000:.2f} mm)')


if __name__ == '__main__':
    main()
