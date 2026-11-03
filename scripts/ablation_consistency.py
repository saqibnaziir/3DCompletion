"""
Does hard data consistency actually earn its place? (Olivier, 2026-08-21 review.)

Three arms, one fixed trained model, one test set:

  (1) 'none'     classical conditional diffusion. The observation reaches the
                 model ONLY through the conditioning channels; the sampler runs
                 free (known_mask all-False) and the output is read out for
                 EVERY vertex, observed ones included. Nothing is replaced.
  (2) 'final'    hard consistency once, at the end: sample free exactly as in
                 (1), then paste the observed vertices back into the result.
  (3) 'perstep'  hard consistency at every DDIM step (RePaint-style overwrite +
                 re-noise), then the same final paste. This is what we do.

Arms (1) and (2) share ONE sampling run: they are the same trajectory read out
two different ways, so the missing region is bit-identical between them and the
comparison is exactly paired with zero sampling noise. Arm (3) is a second run
seeded identically, so it starts from the same x_T.

WHY FOUR METRIC COLUMNS AND NOT ONE
Missing-region L2 alone cannot answer this question, and the earlier run already
showed why (0.00208 vs 0.00211 -- a wash). The observation is fed to the network
either way, so the filled region barely moves. What per-step projection buys is
visible elsewhere:

  * known-region error   -- arm (1) perturbs vertices you actually MEASURED.
                            Arms (2)/(3) are exactly 0 by construction.
  * seam normal error    -- positions at the seam are exact once you paste, so
                            L2 is blind there; a completion arriving at the
                            boundary at the wrong ANGLE is a visible crease that
                            only the normal metric sees. This is where (2) and
                            (3) can genuinely differ: (2) pastes a boundary the
                            sampler never treated as fixed.
  * uncertainty on the   -- with --K > 1, arm (1) has NONZERO spread on observed
    observed region         vertices. Arms (2)/(3) are exactly 0. This is what
                            makes "u = 0 on the observed region" a guarantee of
                            the method rather than a happy accident, and it is
                            the direct link to the calibration section.

Run (GPU box with the trained checkpoint) -- note --param/--tag MUST match the
arm you are reporting, or you silently score a different model:
  python scripts/ablation_consistency.py \
         --config_file config/config_diffusion.cfg \
         --param x0 --tag depthw \
         --test_dir data/completion_test_samples/test_set --K 10
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
from glob import glob
from trimesh import load_mesh

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from utils.normals import normal_angular_error, boundary_band, normal_stats
from utils.boundary_field import BoundaryFieldSolver
from scripts.infer_diffusion import (
    parse_diffusion_section, build_denoiser, get_template_info, load_norm,
)

ARMS = ('none', 'final', 'perstep')
ARM_LABEL = {
    'none':    'classical cond. diffusion (no replacement)',
    'final':   'hard consistency at the end only',
    'perstep': 'hard consistency at every step (ours)',
}


# --------------------------------------------------------------------------- #
class Coder:
    """Original coords <-> the standardized offset space the diffusion lives in.

    Hoisted out of the sampling loop: load_norm() re-reads the normalization
    tensors from disk on every call, which is wasted work when it is the same
    template for every mesh in the run.
    """

    def __init__(self, config, sigma):
        mean, std, mean_only = load_norm(config)
        self.mean = mean.numpy()
        self.std = std.numpy() if std is not None else None
        self.mean_only = mean_only
        self.sigma = sigma.cpu().numpy().astype(np.float32)

    def cond_from(self, verts_np, mask):
        """(N,4) conditioning tensor input: standardized xyz (missing zeroed) + mask."""
        vn = verts_np.astype(np.float32).copy() - self.mean
        if not self.mean_only and self.std is not None:
            vn = vn / self.std
        vn[mask] = 0.0
        cond_xyz = vn / self.sigma
        cond_xyz[mask] = 0.0
        mch = np.ones((len(verts_np), 1), np.float32)
        mch[mask] = 0.0
        return np.concatenate([cond_xyz, mch], 1)

    def decode(self, x0_np):
        """Standardized offset field -> original coordinates."""
        out = x0_np * self.sigma
        if not self.mean_only and self.std is not None:
            out = out * self.std
        return out + self.mean


@torch.no_grad()
def sample_pair(denoiser, diffusion, coder, verts_np, mask, device,
                clip_value, num_steps):
    """One free run and one per-step-projected run from the SAME initial noise.

    Returns (free, projected), each (N,3) in original coords, read out at every
    vertex with nothing pasted -- the caller decides what to restore.

    The shared start point matters: torch.manual_seed is reset to the same value
    before each run so x_T is identical, making the two sampling modes a paired
    comparison rather than two independent draws.
    """
    N = len(verts_np)
    cond = torch.tensor(coder.cond_from(verts_np, mask), device=device)
    real_known = cond[:, 3] > 0.5
    known_values = cond[:, :3].contiguous()

    outs = []
    for km in (torch.zeros_like(real_known), real_known):
        x0 = diffusion.ddim_sample(denoiser, cond, B=1, nodes_per_graph=N,
                                   known_mask=km, known_values=known_values,
                                   num_steps=num_steps, device=device,
                                   clip_value=clip_value)
        outs.append(coder.decode(x0.cpu().numpy()))
    return outs[0], outs[1]


def assemble(arm, free, proj, verts_np, mask):
    """Read out one arm from the two raw sampler outputs."""
    if arm == 'none':
        return free.copy()                       # model's own value everywhere
    src = free if arm == 'final' else proj
    out = verts_np.copy()                        # observed vertices verbatim
    out[mask] = src[mask]
    return out


# --------------------------------------------------------------------------- #
def score_mesh(comp, gt, mask, faces, band):
    """Positional + surface-orientation metrics for one completed mesh."""
    e = np.linalg.norm(comp - gt, axis=1)
    row = {
        'missing_mm': float(e[mask].mean()) * 1000.0,
        'known_mm':   float(e[~mask].mean()) * 1000.0,
        'overall_mm': float(e.mean()) * 1000.0,
        'known_max_mm': float(e[~mask].max()) * 1000.0,
    }
    ang, valid = normal_angular_error(comp, gt, faces)
    row.update(normal_stats(ang, valid, band, 'nrm_seam'))
    row.update(normal_stats(ang, valid, mask, 'nrm_missing'))
    if not band.any():
        # No seam exists: this mask is a whole disconnected component (COMA's
        # eyeballs are separate components, and one test mask is exactly one
        # eyeball). normal_stats reports 0.0 for an empty selection, which would
        # be averaged in as a perfect score. NaN keeps it out of the mean.
        for k in [k for k in row if k.startswith('nrm_seam_') and k != 'nrm_seam_n']:
            row[k] = float('nan')
    return row


def paired_tests(per_mesh, arm_a, arm_b, key):
    """Paired t-test + Wilcoxon of arm_a vs arm_b over meshes; n=20 needs both."""
    from scipy.stats import ttest_rel, wilcoxon
    a = np.array([m[arm_a][key] for m in per_mesh], dtype=float)
    b = np.array([m[arm_b][key] for m in per_mesh], dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))        # drop seamless masks, pairwise
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return {'mean_a': float('nan'), 'mean_b': float('nan'),
                'mean_diff': float('nan'), 'better': 0, 'n': 0,
                'p_ttest': float('nan'), 'p_wilcoxon': float('nan'),
                'identical': False}
    d = a - b
    out = {'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
           'mean_diff': float(d.mean()),
           'better': int((a < b).sum()), 'n': len(d)}
    if np.allclose(d, 0):
        # Identical by construction (e.g. missing region of 'none' vs 'final').
        out['p_ttest'] = 1.0
        out['p_wilcoxon'] = 1.0
        out['identical'] = True
        return out
    out['identical'] = False
    out['p_ttest'] = float(ttest_rel(a, b).pvalue)
    try:
        out['p_wilcoxon'] = float(wilcoxon(a, b).pvalue)
    except ValueError:
        out['p_wilcoxon'] = float('nan')
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--param', default='x0', choices=['eps', 'x0', 'v'],
                    help="prediction target of the arm being scored. MUST match "
                         "how it was trained -- sampling an x0 arm as eps is garbage.")
    ap.add_argument('--tag', default=None,
                    help="checkpoint subfolder suffix -> predict_<param>[_<tag>]/")
    ap.add_argument('--test_dir', default=None)
    ap.add_argument('--gt_dir', default=None, help="default: <test_dir>/gt if present")
    ap.add_argument('--mask_dir', default=None, help="default: <test_dir>/partial if present")
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps override')
    ap.add_argument('--clip', type=float, default=None, help='x0-clamp override')
    ap.add_argument('--K', type=int, default=1,
                    help='samples per mesh; >1 adds the observed-region '
                         'uncertainty column (arm 1 nonzero, arms 2/3 exactly 0)')
    ap.add_argument('--rings', type=int, default=2, help='seam band half-width, in hops')
    ap.add_argument('--seam_band_frac', type=float, default=0.2,
                    help="a mask whose seam band exceeds this fraction of the mesh "
                         "has no localized boundary (scattered holes put a seam "
                         "almost everywhere), so its 'seam' number is really a "
                         "whole-mesh number. Those meshes are reported separately.")
    ap.add_argument('--max_meshes', type=int, default=None, help='limit for a smoke test')
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--use_last', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm_dir = f"predict_{args.param}" + (f"_{args.tag}" if args.tag else "")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm_dir)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  ckpt={arm_dir}  steps={num_steps}  K={args.K}')

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema,
                              use_last=args.use_last)
    # predict= is what decodes the network output; leaving it at the default
    # would decode an x0-trained model as if it emitted eps.
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)

    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces)
    N_t = len(template_mesh.vertices)
    adjacency = BoundaryFieldSolver(faces, np.asarray(template_mesh.vertices))._A

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device); saved_clip = scale.get('clip_value', None)
    else:
        sigma = scale.to(device); saved_clip = None
    clip_value = args.clip if args.clip is not None else saved_clip
    print(f'clip_value={clip_value}')

    coder = Coder(config, sigma)

    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples', 'test_set')
    gt_dir = args.gt_dir or (os.path.join(test_dir, 'gt')
                             if os.path.isdir(os.path.join(test_dir, 'gt')) else test_dir)
    mask_dir = args.mask_dir or (os.path.join(test_dir, 'partial')
                                 if os.path.isdir(os.path.join(test_dir, 'partial')) else test_dir)
    out_dir = args.out_dir or os.path.join(test_dir, 'consistency_ablation')
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if args.max_meshes:
        gt_files = gt_files[:args.max_meshes]
    if not gt_files:
        raise SystemExit(f'no *_gt_complete.ply under {gt_dir}')
    print(f'{len(gt_files)} test meshes  (gt={gt_dir}, masks={mask_dir})\n')

    per_mesh = []
    print(f'{"mesh":10s} {"arm":10s} {"miss mm":>9s} {"known mm":>9s} '
          f'{"seam deg":>9s} {"u_obs mm":>9s}')
    print('-' * 62)

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
        band = boundary_band(adjacency, mask, rings=args.rings)

        # Same seed per mesh -> every arm starts from the same x_T (paired).
        stacks = {a: [] for a in ARMS}
        for k in range(args.K):
            torch.manual_seed(args.seed + 1000 * k)
            np.random.seed(args.seed + 1000 * k)
            free, proj = sample_pair(denoiser, diffusion, coder, partial, mask,
                                     device, clip_value, num_steps)
            for a in ARMS:
                stacks[a].append(assemble(a, free, proj, partial, mask))

        entry = {'mesh': base, 'missing_pct': float(100 * mask.sum() / len(mask)),
                 'n_missing': int(mask.sum()), 'n_band': int(band.sum())}
        for a in ARMS:
            st = np.stack(stacks[a], 0)                       # (K,N,3)
            row = score_mesh(st[0], gt, mask, faces, band)    # single sample
            if args.K > 1:
                u = np.linalg.norm(st.std(axis=0), axis=1)    # per-vertex spread
                row['u_known_mm'] = float(u[~mask].mean()) * 1000.0
                row['u_missing_mm'] = float(u[mask].mean()) * 1000.0
                mu = st.mean(axis=0)
                row['meanK_missing_mm'] = float(
                    np.linalg.norm(mu - gt, axis=1)[mask].mean()) * 1000.0
            entry[a] = row
            print(f'{base:10s} {a:10s} {row["missing_mm"]:9.4f} '
                  f'{row["known_mm"]:9.4f} {row["nrm_seam_deg_mean"]:9.4f} '
                  f'{row.get("u_known_mm", float("nan")):9.4f}')
        per_mesh.append(entry)

    if not per_mesh:
        raise SystemExit('no scorable meshes (missing masks?)')

    # ---------------- aggregate ---------------- #
    keys = ['missing_mm', 'known_mm', 'overall_mm', 'known_max_mm',
            'nrm_seam_deg_mean', 'nrm_seam_deg_p95', 'nrm_missing_deg_mean']
    if args.K > 1:
        keys += ['u_known_mm', 'u_missing_mm', 'meanK_missing_mm']
    # nan-aware: seam metrics are undefined for seamless masks (see score_mesh),
    # so they average over fewer meshes than the positional columns. `n` records
    # how many actually contributed, rather than hiding the difference.
    agg = {}
    for a in ARMS:
        agg[a] = {}
        for k in keys:
            v = np.array([m[a][k] for m in per_mesh], dtype=float)
            good = ~np.isnan(v)
            agg[a][k] = {'mean': float(v[good].mean()) if good.any() else float('nan'),
                         'std': float(v[good].std()) if good.any() else float('nan'),
                         'n': int(good.sum())}

    tests = {}
    for k in ['missing_mm', 'meanK_missing_mm', 'known_mm', 'overall_mm',
              'nrm_seam_deg_mean']:
        if k not in keys:
            continue
        for a, b in [('perstep', 'none'), ('perstep', 'final'), ('final', 'none')]:
            tests[f'{a}_vs_{b}:{k}'] = paired_tests(per_mesh, a, b, k)

    # Seam, restricted to masks that actually have a LOCALIZED boundary. A
    # scattered mask seams almost every vertex (bands of ~4800/5023 here), so
    # pooling it with patch masks measures whole-mesh normals under the name
    # "seam" and dilutes the effect the column exists to detect.
    localized = [m for m in per_mesh
                 if 0 < m['n_band'] <= args.seam_band_frac * N_t]
    seam_loc = {}
    if localized:
        for a in ARMS:
            v = np.array([m[a]['nrm_seam_deg_mean'] for m in localized], float)
            seam_loc[a] = {'mean': float(v.mean()), 'std': float(v.std()),
                           'n': len(v)}
        for a, b in [('perstep', 'none'), ('perstep', 'final'), ('final', 'none')]:
            seam_loc[f'{a}_vs_{b}'] = paired_tests(localized, a, b,
                                                   'nrm_seam_deg_mean')
        seam_loc['meshes'] = [m['mesh'] for m in localized]

    n = len(per_mesh)
    print('\n' + '=' * 78)
    print(f'AGGREGATE over {n} meshes  (K={args.K}, steps={num_steps}, arm={arm_dir})')
    print(f'{"arm":44s} {"miss mm":>9s} {"known mm":>9s} {"seam deg":>9s}')
    print('-' * 78)
    for a in ARMS:
        print(f'{ARM_LABEL[a]:44s} {agg[a]["missing_mm"]["mean"]:9.4f} '
              f'{agg[a]["known_mm"]["mean"]:9.4f} '
              f'{agg[a]["nrm_seam_deg_mean"]["mean"]:9.4f}')
    n_seam = agg['perstep']['nrm_seam_deg_mean']['n']
    if n_seam != n:
        print(f'  (seam column averages {n_seam}/{n} meshes; the rest have a '
              f'fully-disconnected mask and therefore no seam)')
    if args.K > 1:
        print(f'\n{"arm":44s} {"u_obs mm":>9s} {"u_miss mm":>9s}')
        for a in ARMS:
            print(f'{ARM_LABEL[a]:44s} {agg[a]["u_known_mm"]["mean"]:9.4f} '
                  f'{agg[a]["u_missing_mm"]["mean"]:9.4f}')

    if localized:
        print(f'\nseam on the {len(localized)} LOCALIZED-boundary masks only '
              f'(band <= {args.seam_band_frac:.0%} of mesh):')
        for a in ARMS:
            print(f'  {ARM_LABEL[a]:44s} {seam_loc[a]["mean"]:9.4f} deg')
        for a, b in [('perstep', 'none'), ('perstep', 'final'), ('final', 'none')]:
            t = seam_loc[f'{a}_vs_{b}']
            print(f'    {a:8s} vs {b:8s}: diff={t["mean_diff"]:+.4f}  '
                  f'p_t={t["p_ttest"]:.4g}  p_w={t["p_wilcoxon"]:.4g}  '
                  f'wins={t["better"]}/{t["n"]}')

    print('\npaired tests (n={} meshes):'.format(n))
    for name, t in tests.items():
        flag = ' [identical by construction]' if t['identical'] else ''
        print(f'  {name:44s} diff={t["mean_diff"]:+.4f}  '
              f'p_t={t["p_ttest"]:.4g}  p_w={t["p_wilcoxon"]:.4g}  '
              f'wins={t["better"]}/{t["n"]}{flag}')

    # ---------------- save ---------------- #
    meta = {'arm_ckpt': arm_dir, 'param': args.param, 'tag': args.tag,
            'K': args.K, 'steps': num_steps, 'clip_value': clip_value,
            'rings': args.rings, 'seam_band_frac': args.seam_band_frac,
            'seed': args.seed, 'n_meshes': n,
            'gt_dir': gt_dir, 'mask_dir': mask_dir}
    with open(os.path.join(out_dir, 'consistency_ablation.json'), 'w') as f:
        json.dump({'meta': meta, 'aggregate': agg, 'tests': tests,
                   'seam_localized': seam_loc, 'per_mesh': per_mesh}, f, indent=2)

    with open(os.path.join(out_dir, 'consistency_ablation.csv'), 'w') as f:
        f.write('arm,label,' + ','.join(f'{k}_mean,{k}_std,{k}_n' for k in keys)
                + ',n_meshes\n')
        for a in ARMS:
            vals = ','.join(f'{agg[a][k]["mean"]:.6f},{agg[a][k]["std"]:.6f},'
                            f'{agg[a][k]["n"]}' for k in keys)
            f.write(f'{a},"{ARM_LABEL[a]}",{vals},{n}\n')

    with open(os.path.join(out_dir, 'per_mesh.csv'), 'w') as f:
        f.write('mesh,missing_pct,arm,' + ','.join(keys) + '\n')
        for m in per_mesh:
            for a in ARMS:
                f.write(f'{m["mesh"]},{m["missing_pct"]:.2f},{a},'
                        + ','.join(f'{m[a][k]:.6f}' for k in keys) + '\n')

    # paste-ready LaTeX rows (ours is bolded, as in the other tables)
    tex = ['% auto-generated by scripts/ablation_consistency.py',
           f'% {arm_dir}, K={args.K}, {num_steps} steps, {n} meshes']
    for a in ARMS:
        cols = ['{:.3f}'.format(agg[a]['missing_mm']['mean']),
                '{:.3f}'.format(agg[a]['known_mm']['mean']),
                '{:.3f}'.format(agg[a]['nrm_seam_deg_mean']['mean'])]
        if args.K > 1:
            cols.append('{:.3f}'.format(agg[a]['u_known_mm']['mean']))
        if a == 'perstep':
            cols = [r'\mathbf{' + c + '}' for c in cols]
        tex.append(ARM_LABEL[a] + ' & ' + ' & '.join('$' + c + '$' for c in cols)
                   + r' \\')
    with open(os.path.join(out_dir, 'consistency_snippet.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    print(f'\nwrote {out_dir}/consistency_ablation.{{json,csv}}, per_mesh.csv, '
          f'consistency_snippet.tex')


if __name__ == '__main__':
    main()
