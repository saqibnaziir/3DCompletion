"""
How much of the diffusion arm's normal error is IRREDUCIBLE SAMPLING NOISE?

Motivation (2026-07-31). The x0_depthw arm sits at 4.57 deg mean normal error in
the hole, and the error field is ~28x rougher than the surface it sits on --
i.e. it is high-frequency ripple, exactly the pattern independent per-vertex
sampling noise would produce. Before spending two days training a normal loss,
we need to know whether that ripple is MODEL error (a loss can fix it) or
SAMPLING variance (a loss cannot -- the fix would be in the sampler).

The test needs no training. For each test mesh draw K DDIM samples and compare:

    single   = E_k [ normal_error( sample_k ) ]      one draw, averaged over k
    mean_of_K= normal_error( mean_k sample_k )       the empirical conditional mean

Averaging K samples cancels zero-mean independent noise (~1/sqrt(K)) while
leaving systematic error untouched. So:

    mean_of_K << single   ->  VARIANCE floor. A normal loss is the wrong lever;
                              look at the sampler (fewer/better steps, lower eta,
                              averaging, or a variance-reduction trick).
    mean_of_K ~= single   ->  SYSTEMATIC error. The model genuinely mis-orients
                              the surface, and a normal loss targets it directly.

Note the mean-of-K mesh is NOT a valid diffusion sample -- it is the Monte-Carlo
estimate of the conditional mean, i.e. what a perfect regressor would output. It
is used here purely as a variance probe, and averaging is only meaningful
because of dense correspondence (vertex i <-> i across all samples).

Writes everything to its OWN folder (default
data/completion_test_samples/test_set/diag_sampling_floor/) so it cannot mix
with the arm_comparison_* tables or any completed_* mesh folder.

Run (conda 3dmesh python):
  python scripts/diag_sampling_floor.py --data COMA --param x0 --tag depthw \
      --test_dir data/completion_test_samples/test_set --K 10
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import time
import numpy as np
import scipy.sparse as sp
import torch
from glob import glob
from trimesh import Trimesh, load_mesh

from config.config import read_config, resolve_config_file
from utils.diffusion import GaussianDiffusion
from utils.funcs import save_ply_explicit
from utils.normals import normal_angular_error, boundary_band
from utils.boundary_field import BoundaryFieldSolver
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)


def uniform_laplacian(adj, n):
    """L = I - D^-1 A (umbrella). Used only for the roughness ratio."""
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sp.eye(n, format='csr') - sp.diags(1.0 / deg) @ adj


def mesh_stats(verts, gt, faces, mask, band, L):
    """Normal + position + roughness numbers for one candidate mesh."""
    ang, valid = normal_angular_error(verts, gt, faces)
    sel = mask & valid
    pos = np.linalg.norm(verts - gt, axis=1)
    lc = np.linalg.norm(L @ verts, axis=1)
    lg = np.linalg.norm(L @ gt, axis=1)
    bsel = band & valid & ~mask          # seam = known side, positions exact
    return {
        'nrm_deg': float(ang[sel].mean()) if sel.any() else 0.0,
        'nrm_p95': float(np.percentile(ang[sel], 95)) if sel.any() else 0.0,
        'seam_deg': float(ang[bsel].mean()) if bsel.any() else 0.0,
        'pos_mm': float(1000.0 * pos[mask].mean()) if mask.any() else 0.0,
        'rough_ratio': float(lc[mask].mean() / max(lg[mask].mean(), 1e-12))
                       if mask.any() else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--param', default='x0')
    ap.add_argument('--tag', default='depthw')
    ap.add_argument('--test_dir',
                    default=os.path.join('data', 'completion_test_samples', 'test_set'))
    ap.add_argument('--gt_dir', default=None)
    ap.add_argument('--mask_dir', default=None)
    ap.add_argument('--out_dir', default=None,
                    help='default: <test_dir>/diag_sampling_floor')
    ap.add_argument('--K', type=int, default=10, help='samples per mesh')
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps')
    ap.add_argument('--limit', type=int, default=None, help='first N test meshes only')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--use_last', action='store_true',
                    help="load diffusion_denoiser_ema_last.pt (LAST epoch) instead of "
                         "diffusion_denoiser_ema.pt (BEST-EVAL epoch); epoch-matching")
    ap.add_argument('--save_meshes', action='store_true',
                    help='also write the mean-of-K and first-sample plys')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = resolve_config_file(args.data, args.config_file)
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    dcfg['predict'] = args.param
    arm = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"config      : {cfg_path}")
    print(f"arm         : {arm}   (predict={args.param})")
    print(f"device      : {device}")

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema,
                              use_last=args.use_last)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces, dtype=np.int64)
    N = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma, clip_value = scale['sigma'].to(device), scale.get('clip_value', None)
    else:
        sigma, clip_value = scale.to(device), None

    solver = BoundaryFieldSolver(faces, np.asarray(template_mesh.vertices))
    L = uniform_laplacian(solver._A, N)

    gt_dir = args.gt_dir or os.path.join(args.test_dir, 'gt')
    mask_dir = args.mask_dir or os.path.join(args.test_dir, 'partial')
    out_dir = args.out_dir or os.path.join(args.test_dir, 'diag_sampling_floor')
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if args.limit:
        gt_files = gt_files[:args.limit]
    if not gt_files:
        raise SystemExit(f"No '*_gt_complete.ply' in {gt_dir}")

    print(f"out dir     : {out_dir}")
    print(f"K           : {args.K} samples/mesh, {num_steps} DDIM steps")
    print(f"meshes      : {len(gt_files)}\n")

    rows = []
    t_start = time.perf_counter()
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            print(f"  skip {base}: no mask"); continue
        mask = np.load(mask_file)
        gt = np.asarray(load_mesh(gt_file, process=False).vertices, dtype=np.float64)
        if len(gt) != N or len(mask) != N:
            print(f"  skip {base}: size mismatch"); continue

        partial = gt.copy().astype(np.float32)
        partial[mask] = 0.0
        band = boundary_band(solver._A, mask, rings=2)

        t0 = time.perf_counter()
        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value, num_steps=num_steps,
                           num_samples=args.K)
        stack = np.stack([s.astype(np.float64) for s in samples], axis=0)   # (K,N,3)

        # Per-sample stats, then their mean: "what one draw typically gives".
        per = [mesh_stats(s, gt, faces, mask, band, L) for s in stack]
        single = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        single_sd = {k: float(np.std([p[k] for p in per])) for k in per[0]}

        # Empirical conditional mean over the K draws.
        mean_mesh = stack.mean(axis=0)
        mean_mesh[~mask] = gt[~mask]          # known stays exact (it already is)
        meanK = mesh_stats(mean_mesh, gt, faces, mask, band, L)

        # Sampling spread for context: per-vertex std across draws, in mm.
        div_mm = float(1000.0 * np.linalg.norm(stack.std(axis=0), axis=1)[mask].mean())

        row = {'sample': base, 'missing_pct': 100.0 * mask.sum() / len(mask),
               'K': args.K, 'diversity_mm': div_mm,
               'single_nrm_deg': single['nrm_deg'], 'single_nrm_sd': single_sd['nrm_deg'],
               'meanK_nrm_deg': meanK['nrm_deg'],
               'single_seam_deg': single['seam_deg'], 'meanK_seam_deg': meanK['seam_deg'],
               'single_pos_mm': single['pos_mm'], 'meanK_pos_mm': meanK['pos_mm'],
               'single_rough': single['rough_ratio'], 'meanK_rough': meanK['rough_ratio'],
               'nrm_drop_pct': 100.0 * (single['nrm_deg'] - meanK['nrm_deg'])
                               / max(single['nrm_deg'], 1e-12),
               'pos_drop_pct': 100.0 * (single['pos_mm'] - meanK['pos_mm'])
                               / max(single['pos_mm'], 1e-12)}
        rows.append(row)

        if args.save_meshes:
            for sub, v in (('mean_of_K', mean_mesh), ('single', stack[0])):
                d = os.path.join(out_dir, sub)
                os.makedirs(d, exist_ok=True)
                save_ply_explicit(Trimesh(vertices=v, faces=faces, process=False),
                                  os.path.join(d, f'{base}_completed_diffusion.ply'))

        print(f"  {base}: normals {single['nrm_deg']:.3f} -> {meanK['nrm_deg']:.3f} deg "
              f"({row['nrm_drop_pct']:+.1f}%)   position {single['pos_mm']:.3f} -> "
              f"{meanK['pos_mm']:.3f} mm ({row['pos_drop_pct']:+.1f}%)   "
              f"[{time.perf_counter()-t0:.1f}s]")

    if not rows:
        print("Nothing scored."); return

    csv_path = os.path.join(out_dir, 'diag_sampling_floor.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def agg(k):
        return float(np.mean([r[k] for r in rows]))

    n_s, n_m = agg('single_nrm_deg'), agg('meanK_nrm_deg')
    p_s, p_m = agg('single_pos_mm'), agg('meanK_pos_mm')
    s_s, s_m = agg('single_seam_deg'), agg('meanK_seam_deg')
    drop = 100.0 * (n_s - n_m) / max(n_s, 1e-12)

    W = 78
    print("\n" + "=" * W)
    print(f"SAMPLING-VARIANCE FLOOR   (K={args.K}, {len(rows)} meshes, arm={arm})")
    print("-" * W)
    print(f"{'quantity':<26} {'1 sample':>12} {'mean of K':>12} {'change':>12}")
    print(f"{'normal err, hole (deg)':<26} {n_s:>12.3f} {n_m:>12.3f} "
          f"{100.0*(n_m-n_s)/max(n_s,1e-12):>11.1f}%")
    print(f"{'normal err, seam (deg)':<26} {s_s:>12.3f} {s_m:>12.3f} "
          f"{100.0*(s_m-s_s)/max(s_s,1e-12):>11.1f}%")
    print(f"{'position err, hole (mm)':<26} {p_s:>12.3f} {p_m:>12.3f} "
          f"{100.0*(p_m-p_s)/max(p_s,1e-12):>11.1f}%")
    print(f"{'roughness |L(v)|/|L(gt)|':<26} {agg('single_rough'):>12.3f} "
          f"{agg('meanK_rough'):>12.3f}")
    print(f"{'sampling spread (mm)':<26} {agg('diversity_mm'):>12.3f}")
    print("-" * W)
    # Thresholds are judgement calls, stated explicitly so the read is not
    # smuggled into the code: averaging K draws cancels independent noise by
    # ~1/sqrt(K), so a large drop means the error was mostly noise.
    if drop >= 40.0:
        verdict = ("VARIANCE-DOMINATED. Most of the orientation error is sampling "
                   "noise.\n  A normal LOSS is the wrong lever -- attack the sampler "
                   "(steps, eta,\n  averaging/variance reduction) instead.")
    elif drop >= 15.0:
        verdict = ("MIXED. A real variance component, but a systematic floor too.\n"
                   "  A normal loss can help, but part of the gap is unreachable by it;\n"
                   "  expect the loss to buy at most the systematic share.")
    else:
        verdict = ("SYSTEMATIC. Averaging barely helps, so the model genuinely\n"
                   "  mis-orients the surface. A normal loss targets exactly this --\n"
                   "  proceed with the training experiment.")
    print(f"normal error drops {drop:.1f}% when averaging {args.K} draws")
    print(f"VERDICT: {verdict}")
    print("=" * W)
    print(f"\nwrote {csv_path}")
    print(f"total {time.perf_counter()-t_start:.0f}s")


if __name__ == '__main__':
    main()
