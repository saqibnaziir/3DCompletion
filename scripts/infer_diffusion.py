"""
Inference / evaluation for diffusion-based mesh completion (Option B).

Sampling = DDIM with hard known-region clamping (RePaint-style), so the known
vertices are preserved EXACTLY and the boundary is seamless by construction
(no post-processing needed).

Modes:
  --eval_test : run on the saved completion_test_samples (GT + mask), compute
                per-vertex error, and (with --num_samples > 1) report diversity.
  --input     : complete a single partial mesh (.ply, same topology as template).

Run:
  python scripts/infer_diffusion.py --config_file config/config_diffusion.cfg --eval_test
  python scripts/infer_diffusion.py --config_file config/config_diffusion.cfg \
         --input data/completion_test_samples/test_000_partial_*.ply --output out.ply
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import configparser
import numpy as np
import torch
from glob import glob
from trimesh import Trimesh, load_mesh

from config.config import read_config, resolve_config_file
from utils.funcs import get_mesh_matrices, save_ply_explicit, get_edge_index
from utils.diffusion import GaussianDiffusion
from utils.diffusion_model import GraphUNetDenoiser
from utils.postprocess import postprocess_completion


# ----------------------------------------------------------------------------
def parse_diffusion_section(config_file):
    cp = configparser.RawConfigParser()
    cp.read(config_file)
    sec = 'diffusion parameters'
    return {
        'timesteps': cp.getint(sec, 'timesteps', fallback=1000),
        'schedule': cp.get(sec, 'schedule', fallback='cosine'),
        'base_channels': cp.getint(sec, 'base_channels', fallback=64),
        'channel_mults': tuple(int(x) for x in
                               cp.get(sec, 'channel_mults', fallback='1,2,4,4').split(',')),
        'time_embed_dim': cp.getint(sec, 'time_embed_dim', fallback=128),
        'sampling_steps': cp.getint(sec, 'sampling_steps', fallback=50),
    }


def load_norm(config):
    norm_path = os.path.join(config['dataset_dir'], 'norm_completion.pt')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(config['dataset_dir'], 'norm.pt')
    nd = torch.load(norm_path, map_location='cpu', weights_only=False)
    return nd['mean'], nd['std'], nd['std'] is None


def build_denoiser(config, dcfg, device, use_ema=True, use_last=False):
    """
    use_last: load 'diffusion_denoiser_ema_last.pt' (the LAST epoch) instead of
    'diffusion_denoiser_ema.pt' (the BEST-EVAL epoch, which is the default and
    is usually NOT the last epoch). Two arms hit best-eval at different epochs,
    so scoring both "as-is" silently compares different amounts of training --
    use this flag to epoch-match instead.
    """
    pA, pD, pU = get_mesh_matrices(config)
    denoiser = GraphUNetDenoiser(
        list(pA), list(pD), list(pU),
        in_ch=7, out_ch=3,
        base_channels=dcfg['base_channels'],
        channel_mults=dcfg['channel_mults'],
        time_embed_dim=dcfg['time_embed_dim'],
    ).to(device)

    ckpt_dir = config['checkpoint_dir']
    ema_path = os.path.join(ckpt_dir, 'diffusion_denoiser_ema.pt')
    last_path = os.path.join(ckpt_dir, 'diffusion_denoiser_ema_last.pt')
    raw_path = os.path.join(ckpt_dir, 'diffusion_denoiser.pt')
    if use_last:
        if not use_ema:
            raise SystemExit("--use_last applies to the EMA checkpoint; drop --no_ema.")
        if not os.path.exists(last_path):
            raise SystemExit(f"--use_last given but {last_path} does not exist.")
        path = last_path
    else:
        path = ema_path if (use_ema and os.path.exists(ema_path)) else raw_path
    print(f"Loading denoiser weights: {path}")
    sd = torch.load(path, map_location=device, weights_only=False)
    denoiser.load_state_dict(sd)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False
    return denoiser


def get_template_info(config):
    template_mesh = load_mesh(config['template_file'], process=False)
    ei_path = os.path.join(config['dataset_dir'], 'processed_completion', 'edge_index.pt')
    if os.path.exists(ei_path):
        edge_index = torch.load(ei_path, weights_only=False)
    else:
        edge_index = get_edge_index(template_mesh.vertices, template_mesh.faces)
    return template_mesh, edge_index


# ----------------------------------------------------------------------------
@torch.no_grad()
def complete(denoiser, diffusion, verts_np, missing_mask, config, device,
             sigma, clip_value=None, num_steps=50, num_samples=1):
    """
    Returns a list of `num_samples` completed (N,3) meshes in ORIGINAL coords.
    Known vertices are exact; missing vertices come from the diffusion sample.

    `sigma` (per-axis std) standardizes the offset field to unit variance — must
    match the value used during training (loaded from diffusion_scale.pt).
    """
    mean, std, mean_only = load_norm(config)
    mean_np = mean.numpy()
    std_np = std.numpy() if std is not None else None
    sigma_np = sigma.cpu().numpy().astype(np.float32)

    N = len(verts_np)
    verts_norm = verts_np.astype(np.float32).copy() - mean_np
    if not mean_only and std_np is not None:
        verts_norm = verts_norm / std_np
    verts_norm[missing_mask] = 0.0

    # Standardize to the diffusion working space.
    cond_xyz = verts_norm / sigma_np
    cond_xyz[missing_mask] = 0.0

    mask_channel = np.ones((N, 1), dtype=np.float32)
    mask_channel[missing_mask] = 0.0
    cond_np = np.concatenate([cond_xyz, mask_channel], axis=1)        # (N, 4)

    cond1 = torch.tensor(cond_np, device=device)
    known_mask1 = (cond1[:, 3] > 0.5)
    known_values1 = cond1[:, :3].contiguous()

    results = []
    for _ in range(num_samples):
        x0 = diffusion.ddim_sample(
            denoiser, cond1, B=1, nodes_per_graph=N,
            known_mask=known_mask1, known_values=known_values1,
            num_steps=num_steps, device=device, progress=(num_samples == 1),
            clip_value=clip_value)

        # Un-standardize -> normalized offset -> original coordinates.
        out = x0.cpu().numpy() * sigma_np
        if not mean_only and std_np is not None:
            out = out * std_np
        out = out + mean_np

        final = verts_np.copy()
        final[missing_mask] = out[missing_mask]      # blend (known exact)
        results.append(final)
    return results


# ----------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(denoiser, diffusion, verts_np, config, device, sigma,
                clip_value=None, num_steps=50, num_samples=1):
    """
    R1 reconstruction ("regenerate"): condition on the FULL mesh (every vertex
    known, mask=1 everywhere) and regenerate the entire offset field from noise
    WITHOUT hard clamping. Unlike completion (which copies known vertices back
    exactly), this reads out the model's own reproduction of all N vertices, so
    the error measures how faithfully the diffusion model reconstructs geometry.

    Returns a list of `num_samples` reconstructed (N,3) meshes in ORIGINAL coords.
    """
    mean, std, mean_only = load_norm(config)
    mean_np = mean.numpy()
    std_np = std.numpy() if std is not None else None
    sigma_np = sigma.cpu().numpy().astype(np.float32)

    N = len(verts_np)
    verts_norm = verts_np.astype(np.float32).copy() - mean_np
    if not mean_only and std_np is not None:
        verts_norm = verts_norm / std_np
    cond_xyz = verts_norm / sigma_np                    # full mesh, nothing zeroed

    mask_channel = np.ones((N, 1), dtype=np.float32)    # all vertices known
    cond_np = np.concatenate([cond_xyz, mask_channel], axis=1)        # (N, 4)

    cond1 = torch.tensor(cond_np, device=device)
    # No clamp: known_mask all-False so the sampler regenerates EVERY vertex
    # (the conditioning coords still guide the denoiser via the cond channels).
    known_mask1 = torch.zeros(N, dtype=torch.bool, device=device)
    known_values1 = cond1[:, :3].contiguous()

    results = []
    for _ in range(num_samples):
        x0 = diffusion.ddim_sample(
            denoiser, cond1, B=1, nodes_per_graph=N,
            known_mask=known_mask1, known_values=known_values1,
            num_steps=num_steps, device=device, progress=(num_samples == 1),
            clip_value=clip_value)

        out = x0.cpu().numpy() * sigma_np
        if not mean_only and std_np is not None:
            out = out * std_np
        out = out + mean_np
        results.append(out)                              # full reconstruction
    return results


def eval_recon(denoiser, diffusion, config, device, template_mesh, sigma,
               clip_value, num_steps, num_samples, test_dir=None):
    """Reconstruct every GT mesh in the test set and report per-vertex L2."""
    if test_dir is None:
        test_dir = os.path.join(config['dataset_dir'], 'completion_test_samples')
    out_dir = os.path.join(test_dir, 'reconstructed_diffusion')
    os.makedirs(out_dir, exist_ok=True)
    faces = np.array(template_mesh.faces)
    N_t = len(template_mesh.vertices)

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    print(f"Reconstructing {len(gt_files)} test meshes "
          f"(R1: full-mesh conditioning, no clamp, DDIM steps={num_steps})\n")

    recon_errs = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t:
            print(f"  skip {base}: size mismatch")
            continue

        samples = reconstruct(denoiser, diffusion, gt, config, device, sigma,
                              clip_value=clip_value, num_steps=num_steps,
                              num_samples=num_samples)
        recon = samples[0]
        err = np.linalg.norm(recon - gt, axis=1)
        r_err = err.mean()
        recon_errs.append(r_err)

        save_ply_explicit(Trimesh(vertices=recon, faces=faces, process=False),
                          os.path.join(out_dir, f'{base}_reconstructed.ply'))
        print(f"  {base}: reconstruction L2 = {r_err:.6f}")

    if recon_errs:
        print("\n" + "=" * 60)
        print(f"AGGREGATE over {len(recon_errs)} meshes")
        print(f"  reconstruction error: {np.mean(recon_errs):.6f} "
              f"+/- {np.std(recon_errs):.6f}")
        print(f"  meshes saved to: {out_dir}")


# ----------------------------------------------------------------------------
def eval_test(denoiser, diffusion, config, device, template_mesh, sigma, clip_value,
              num_steps, num_samples, smooth_cfg=None, test_dir=None,
              out_suffix='', gt_dir=None, mask_dir=None):
    if test_dir is None:
        test_dir = os.path.join(config['dataset_dir'], 'completion_test_samples')
    # out_suffix keeps arms from overwriting each other's meshes (default '' =
    # the original path, so existing workflows are unchanged).
    out_dir = os.path.join(test_dir, f'completed_diffusion{out_suffix}')
    os.makedirs(out_dir, exist_ok=True)
    faces = np.array(template_mesh.faces)
    N_t = len(template_mesh.vertices)

    # Two layouts in the wild: the original flat one (everything directly in
    # test_dir) and the reorganized test_set (gt/ + partial/ subfolders, which
    # compare_arms.py already assumes). Auto-detect, allow explicit override.
    if gt_dir is None:
        gt_dir = os.path.join(test_dir, 'gt')
        if not os.path.isdir(gt_dir):
            gt_dir = test_dir
    if mask_dir is None:
        mask_dir = os.path.join(test_dir, 'partial')
        if not os.path.isdir(mask_dir):
            mask_dir = test_dir

    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if not gt_files:
        raise SystemExit(
            f"No '*_gt_complete.ply' found in {gt_dir}.\n"
            f"  Pass --gt_dir / --mask_dir explicitly if your test set is laid out "
            f"differently.\n"
            f"  (Refusing to report a successful run over zero samples.)")
    print(f"GT dir   : {gt_dir}")
    print(f"Mask dir : {mask_dir}")
    print(f"Evaluating {len(gt_files)} test samples "
          f"(DDIM steps={num_steps}, samples/mesh={num_samples}, "
          f"smooth={smooth_cfg is not None})\n")

    miss_errs, overall_errs, miss_pcts, diversities = [], [], [], []
    skipped = []

    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            skipped.append(base)
            continue
        mask = np.load(mask_file)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t or len(mask) != N_t:
            print(f"  skip {base}: size mismatch")
            continue

        partial = gt.copy()
        partial[mask] = 0.0

        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value,
                           num_steps=num_steps, num_samples=num_samples)

        # Use the first sample for the error metric; save it.
        comp = samples[0]
        if smooth_cfg is not None:
            comp, _ = postprocess_completion(comp, faces, mask, **smooth_cfg)
        err = np.linalg.norm(comp - gt, axis=1)
        m_err = err[mask].mean() if mask.any() else 0.0
        o_err = err.mean()
        miss_errs.append(m_err)
        overall_errs.append(o_err)
        miss_pcts.append(100.0 * mask.sum() / len(mask))

        save_ply_explicit(Trimesh(vertices=comp, faces=faces, process=False),
                          os.path.join(out_dir, f'{base}_completed_diffusion.ply'))

        div_str = ""
        if num_samples > 1:
            stack = np.stack(samples, axis=0)               # (K, N, 3)
            # mean per-vertex std over the missing region = sample diversity
            div = np.linalg.norm(stack.std(axis=0), axis=1)[mask].mean() if mask.any() else 0.0
            diversities.append(div)
            div_str = f"  diversity={div:.6f}"
            for k, s in enumerate(samples):
                save_ply_explicit(Trimesh(vertices=s, faces=faces, process=False),
                                  os.path.join(out_dir, f'{base}_sample{k:02d}.ply'))

        print(f"  {base}: missing={m_err:.6f}  overall={o_err:.6f}  "
              f"({miss_pcts[-1]:.1f}% missing){div_str}")

    if skipped:
        print(f"\n  WARNING: {len(skipped)} sample(s) had no *_mask.npy in {mask_dir} "
              f"and were skipped: {', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}")

    if miss_errs:
        print("\n" + "=" * 60)
        print(f"AGGREGATE over {len(miss_errs)} samples")
        print(f"  avg missing %:      {np.mean(miss_pcts):.1f}%")
        print(f"  missing error:      {np.mean(miss_errs):.6f} +/- {np.std(miss_errs):.6f}")
        print(f"  overall error:      {np.mean(overall_errs):.6f} +/- {np.std(overall_errs):.6f}")
        if diversities:
            print(f"  sample diversity:   {np.mean(diversities):.6f}")
        print(f"  meshes saved to: {out_dir}")


def complete_single(denoiser, diffusion, config, device, template_mesh, sigma, clip_value,
                    input_path, output_path, num_steps, num_samples, smooth_cfg=None):
    mesh = load_mesh(input_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float32)
    N_t = len(template_mesh.vertices)
    if len(verts) != N_t:
        print(f"  ERROR: {input_path} has {len(verts)} verts, expected {N_t}")
        return
    missing = np.linalg.norm(verts, axis=1) < 1e-6
    print(f"  Missing: {missing.sum()}/{len(verts)} ({100*missing.sum()/len(verts):.1f}%)")
    samples = complete(denoiser, diffusion, verts, missing, config, device,
                       sigma, clip_value=clip_value,
                       num_steps=num_steps, num_samples=num_samples)
    faces = np.array(template_mesh.faces)
    comp0 = samples[0]
    if smooth_cfg is not None:
        comp0, _ = postprocess_completion(comp0, faces, missing, **smooth_cfg)
    if output_path is None:
        output_path = input_path.replace('.ply', '_completed_diffusion.ply')
    save_ply_explicit(Trimesh(vertices=comp0, faces=faces, process=False), output_path)
    print(f"  Saved: {output_path}")
    for k in range(1, num_samples):
        p = output_path.replace('.ply', f'_sample{k:02d}.ply')
        save_ply_explicit(Trimesh(vertices=samples[k], faces=faces, process=False), p)


def main():
    parser = argparse.ArgumentParser(description='Diffusion mesh completion inference')
    parser.add_argument("--config_file", type=str, default=None,
                        help="explicit config path (overrides --data)")
    parser.add_argument("--data", type=str, default=None, choices=['COMA', 'DFAUST'],
                        help="dataset preset: selects config/config_diffusion[_dfaust].cfg")
    parser.add_argument("--eval_test", action='store_true')
    parser.add_argument("--recon", action='store_true',
                        help="R1 reconstruction: condition on the FULL mesh and "
                             "regenerate all vertices from noise (no clamp). "
                             "Reports per-vertex L2 on the test set.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="folder with test_*_gt_complete.ply + masks (default: "
                             "data/completion_test_samples)")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="override DDIM steps")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="samples per mesh (>1 to measure diversity)")
    parser.add_argument("--no_ema", action='store_true', help="use raw (non-EMA) weights")
    parser.add_argument("--use_last", action='store_true',
                        help="load diffusion_denoiser_ema_last.pt (LAST epoch) instead "
                             "of diffusion_denoiser_ema.pt (BEST-EVAL epoch). Use this to "
                             "epoch-match two arms.")
    parser.add_argument("--clip", type=float, default=None,
                        help="override sampling clamp (standardized std units); "
                             "lower = fewer spike/crater outliers (try 6-8)")
    parser.add_argument("--smooth", action='store_true',
                        help="light boundary-aware smoothing on the filled region")
    parser.add_argument("--smooth_method", type=str, default='taubin',
                        choices=['taubin', 'laplacian', 'stitch', 'stitch+taubin'])
    parser.add_argument("--smooth_rings", type=int, default=3)
    parser.add_argument("--smooth_iters", type=int, default=10)
    parser.add_argument("--param", type=str, default=None, choices=['eps', 'x0', 'v'],
                        help="prediction target; overrides config + selects the "
                             "predict_<param> checkpoint subfolder")
    parser.add_argument("--tag", type=str, default=None,
                        help="checkpoint subfolder suffix, matching train_diffusion.py "
                             "--tag (e.g. --param x0 --tag depthw -> predict_x0_depthw)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="folder holding *_gt_complete.ply (default: <test_dir>/gt "
                             "if it exists, else <test_dir>)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="folder holding *_mask.npy (default: <test_dir>/partial "
                             "if it exists, else <test_dir>)")
    parser.add_argument("--out_suffix", type=str, default='',
                        help="suffix for the output mesh folder: writes to "
                             "completed_diffusion<suffix>/ so arms don't overwrite "
                             "each other (e.g. --out_suffix _x0_depthw)")
    args = parser.parse_args()

    cfg_path = resolve_config_file(args.data, args.config_file)
    print(f"Using config: {cfg_path}")
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    if args.param is not None:
        dcfg['predict'] = args.param
    # Match the per-arm checkpoint subfolder written by train_diffusion.py.
    arm = f"predict_{dcfg['predict']}"
    if args.tag:
        arm = f"{arm}_{args.tag}"
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Prediction target: {dcfg['predict']}")

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema,
                              use_last=args.use_last)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=dcfg['predict']).to(device)
    template_mesh, _ = get_template_info(config)
    print(f"Template: {len(template_mesh.vertices)} vertices, {len(template_mesh.faces)} faces")

    scale_path = os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt')
    if not os.path.exists(scale_path):
        raise FileNotFoundError(
            f"Missing {scale_path}. Retrain so the diffusion target scale is saved.")
    scale = torch.load(scale_path, map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device)
        clip_value = scale.get('clip_value', None)
    else:                                    # legacy: bare sigma tensor
        sigma = scale.to(device)
        clip_value = None
    if args.clip is not None:                # CLI override (tighter clamp)
        clip_value = args.clip
    print(f"Diffusion target std (sigma): {sigma.tolist()}")
    print(f"Sampling clip_value: {clip_value}")

    smooth_cfg = None
    if args.smooth:
        smooth_cfg = dict(method=args.smooth_method, num_rings=args.smooth_rings,
                          iterations=args.smooth_iters)
        print(f"Post-smoothing: {smooth_cfg}")
    print()

    if args.recon:
        eval_recon(denoiser, diffusion, config, device, template_mesh,
                   sigma, clip_value, num_steps, args.num_samples,
                   test_dir=args.test_dir)
    if args.eval_test:
        eval_test(denoiser, diffusion, config, device, template_mesh,
                  sigma, clip_value, num_steps, args.num_samples, smooth_cfg=smooth_cfg,
                  test_dir=args.test_dir, out_suffix=args.out_suffix,
                  gt_dir=args.gt_dir, mask_dir=args.mask_dir)
    if args.input:
        for path in sorted(glob(args.input)):
            print(f"Completing {path}")
            complete_single(denoiser, diffusion, config, device, template_mesh,
                            sigma, clip_value, path, args.output, num_steps,
                            args.num_samples, smooth_cfg=smooth_cfg)


if __name__ == "__main__":
    main()
