"""
Paper-revision measurement 1: hole error split by GEODESIC DEPTH decile.

Reuses, unmodified, the two pieces the depth-ramp loss already relies on:
  * utils.boundary_field.BoundaryFieldSolver.depth()  -> d_i, the geodesic
    distance from each missing vertex to the nearest KNOWN boundary vertex,
    with travel confined to the hole (+ its ring).
  * scripts.infer_diffusion.complete()               -> DDIM sampling.

Protocol is byte-for-byte the one that produced the numbers in
data/.../ablation_ladder.csv (research/diag_sampling_floor.py):
    torch.manual_seed(0); np.random.seed(0)
    K=10 draws per mesh, 50 DDIM steps, EMA(best-eval) weights
    per-vertex error e_i = E_k || sample_k,i - gt_i ||     ("single", not mean-of-K)
so `mean over all missing vertices` must reproduce single_pos_mm exactly.

DECILES are taken PER MESH (rank-based on d_i within that mesh's missing set),
then the per-mesh means are averaged over meshes -- the same aggregation the
headline number uses, and what makes the paired test across meshes well defined.

Unreachable missing vertices (COMA's template is 3 components: face + 2
eyeballs, so a fully-missing eyeball has no known neighbour) come back as
d=+inf. depth_weight() already treats those as maximally deep; the ranking here
inherits that, and the count is reported so the choice is visible.

Read-only: loads checkpoints, writes only CSV/JSON under --out_dir.

Run (conda env: 3dmesh)
  python research/revision/rev_depth_deciles.py
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
from scipy import stats
from trimesh import load_mesh

from config.config import read_config, resolve_config_file
from utils.diffusion import GaussianDiffusion
from utils.boundary_field import BoundaryFieldSolver
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)

# label -> (param, tag). The four arms named in the revision request.
ARMS = [
    ('eps',        'eps', None),
    ('x0',         'x0',  None),
    ('x0+depth',   'x0',  'depthw'),
    ('x0+depth+nrm', 'x0', 'depthw_nrm_scratch'),
]


def decile_idx(d, frac=0.1):
    """Rank-based deepest / shallowest decile within one mesh's missing set.

    Rank-based rather than np.quantile so that +inf depths (unreachable hole
    components) sort cleanly to the deep end instead of poisoning a threshold.
    """
    n = len(d)
    k = max(1, int(round(frac * n)))
    order = np.argsort(d, kind='stable')      # ascending; +inf lands last
    return order[-k:], order[:k], k


def run_arm(label, param, tag, config_base, dcfg, device, template_mesh, solver,
            cases, K, steps, seed):
    config = dict(config_base)
    arm = f"predict_{param}" + (f"_{tag}" if tag else "")
    config['checkpoint_dir'] = os.path.join(config_base['checkpoint_dir'], arm)

    d2 = dict(dcfg); d2['predict'] = param
    denoiser = build_denoiser(config, d2, device, use_ema=True)
    diffusion = GaussianDiffusion(timesteps=d2['timesteps'],
                                  schedule=d2['schedule'],
                                  predict=param).to(device)
    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma = scale['sigma'].to(device) if isinstance(scale, dict) else scale.to(device)
    clip_value = scale.get('clip_value', None) if isinstance(scale, dict) else None

    # same seed protocol as diag_sampling_floor.py: seed once, then stream
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = []
    for base, partial, mask, gt in cases:
        t0 = time.perf_counter()
        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value, num_steps=steps,
                           num_samples=K)
        stack = np.stack([s.astype(np.float64) for s in samples], axis=0)  # (K,N,3)

        # per-vertex error of ONE draw, averaged over draws (== 'single')
        e_per_draw = np.linalg.norm(stack - gt[None], axis=2)              # (K,N)
        e = e_per_draw.mean(axis=0)                                        # (N,)

        d_full, _ = solver.depth(mask)
        dm = d_full[mask]
        em = e[mask] * 1000.0                                              # mm
        finite = np.isfinite(dm)

        # DEGENERATE MASK. COMA's template is 3 components (face 3931 + two
        # 546-vertex eyeballs). Three test masks are exactly one whole eyeball,
        # so the hole has NO known neighbour: the boundary ring is empty and
        # every d_i is +inf. depth_weight() calls that "maximally deep" and
        # flattens the ramp, which is right for a LOSS but leaves the decile
        # split undefined -- there is no deep-vs-shallow to measure. Such meshes
        # keep their err_all_mm (so the headline number still covers all 20) and
        # report NaN deciles, and are dropped from the decile aggregate.
        degenerate = not finite.any()
        if degenerate:
            deep = shallow = np.array([], dtype=int)
            k = 0
        else:
            deep, shallow, k = decile_idx(dm)

        nan = float('nan')
        rows.append({
            'arm': label, 'sample': base, 'K': K,
            'n_missing': int(mask.sum()), 'n_decile': int(k),
            'n_inf_depth': int((~finite).sum()),
            'degenerate_depth': int(degenerate),
            'missing_pct': 100.0 * mask.sum() / len(mask),
            'depth_max_mm': float(1000.0 * dm[finite].max()) if finite.any() else nan,
            'err_all_mm': float(em.mean()),
            'err_deep_mm': nan if degenerate else float(em[deep].mean()),
            'err_shallow_mm': nan if degenerate else float(em[shallow].mean()),
            'depth_deep_mm': nan if degenerate else
                             (float(1000.0 * np.mean(dm[deep][np.isfinite(dm[deep])]))
                              if np.isfinite(dm[deep]).any() else nan),
            'depth_shallow_mm': nan if degenerate else float(1000.0 * dm[shallow].mean()),
        })
        tag_d = '  [DEGENERATE depth: whole disconnected component]' if degenerate else ''
        print(f"  [{label:<13}] {base}: all={rows[-1]['err_all_mm']:.3f}  "
              f"deep={rows[-1]['err_deep_mm']:.3f}  "
              f"shallow={rows[-1]['err_shallow_mm']:.3f} mm  "
              f"[{time.perf_counter()-t0:.1f}s]{tag_d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--test_dir',
                    default=os.path.join('data', 'completion_test_samples', 'test_set'))
    ap.add_argument('--K', type=int, default=10,
                    help='draws per mesh; 10 reproduces ablation_ladder.csv')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frac', type=float, default=0.1)
    ap.add_argument('--out_dir', default=os.path.join('data', '_revision'))
    args = ap.parse_args()

    cfg_path = resolve_config_file(args.data, args.config_file)
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces, dtype=np.int64)
    N = len(template_mesh.vertices)
    solver = BoundaryFieldSolver(faces, np.asarray(template_mesh.vertices),
                                 backend='dijkstra')

    gt_dir = os.path.join(args.test_dir, 'gt')
    mask_dir = os.path.join(args.test_dir, 'partial')
    cases = []
    for gt_file in sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply'))):
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mf = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mf):
            continue
        mask = np.load(mf)
        gt = np.asarray(load_mesh(gt_file, process=False).vertices, dtype=np.float64)
        if len(gt) != N or len(mask) != N or not mask.any():
            continue
        partial = gt.copy().astype(np.float32)
        partial[mask] = 0.0
        cases.append((base, partial, mask, gt))

    print(f"device : {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"meshes : {len(cases)} | N={N} | K={args.K} | {steps} DDIM steps | "
          f"decile frac={args.frac}\n")

    all_rows = []
    for label, param, tag in ARMS:
        arm_dir = os.path.join(config['checkpoint_dir'],
                               f"predict_{param}" + (f"_{tag}" if tag else ""))
        if not os.path.exists(os.path.join(arm_dir, 'diffusion_denoiser_ema.pt')):
            print(f"=== SKIP {label}: no EMA checkpoint in {arm_dir}\n")
            continue
        print(f"=== arm {label}  <- {arm_dir}")
        all_rows.extend(run_arm(label, param, tag, config, dcfg, device,
                                template_mesh, solver, cases, args.K, steps,
                                args.seed))
        print()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'depth_deciles_per_mesh.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    # ---- aggregate ----
    labels = [l for l, _, _ in ARMS if any(r['arm'] == l for r in all_rows)]
    per = {l: [r for r in all_rows if r['arm'] == l] for l in labels}
    samples = sorted(set(r['sample'] for r in all_rows))
    degen = sorted({r['sample'] for r in all_rows if r['degenerate_depth']})
    ok = [s for s in samples if s not in degen]

    def vec(l, key, subset):
        m = {r['sample']: r[key] for r in per[l]}
        return np.array([m[s] for s in subset], dtype=np.float64)

    W = 96
    print("=" * W)
    print(f"HOLE ERROR BY GEODESIC-DEPTH DECILE  (mm, K={args.K}, mean of per-mesh means)")
    print(f"  'all missing' over all {len(samples)} meshes; deciles over the "
          f"{len(ok)} meshes with a defined depth field")
    if degen:
        print(f"  excluded from deciles (hole = whole disconnected component, "
              f"d undefined): {', '.join(degen)}")
    print("-" * W)
    print(f"{'arm':<15}{'all (n=%d)' % len(samples):>13}{'all (n=%d)' % len(ok):>13}"
          f"{'deepest 10%':>14}{'shallowest 10%':>17}{'deep/shallow':>14}")
    summary = {}
    for l in labels:
        a20 = vec(l, 'err_all_mm', samples)
        a = vec(l, 'err_all_mm', ok)
        d = vec(l, 'err_deep_mm', ok)
        s = vec(l, 'err_shallow_mm', ok)
        summary[l] = {
            'all_mm_n%d' % len(samples): float(a20.mean()),
            'all_sd_n%d' % len(samples): float(a20.std(ddof=1)),
            'all_mm': float(a.mean()), 'all_sd': float(a.std(ddof=1)),
            'deep_mm': float(d.mean()), 'deep_sd': float(d.std(ddof=1)),
            'shallow_mm': float(s.mean()), 'shallow_sd': float(s.std(ddof=1)),
            'ratio': float(d.mean() / s.mean())}
        print(f"{l:<15}{a20.mean():>13.3f}{a.mean():>13.3f}{d.mean():>14.3f}"
              f"{s.mean():>17.3f}{d.mean()/s.mean():>13.2f}x")
    print("=" * W)

    # ---- paired test: deepest decile, x0 vs x0+depth ----
    tests = {}
    if 'x0' in per and 'x0+depth' in per:
        a = vec('x0+depth', 'err_deep_mm', ok)
        b = vec('x0', 'err_deep_mm', ok)
        t, p_t = stats.ttest_rel(a, b)
        try:
            p_w = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            p_w = float('nan')
        d_ = a - b
        tests['deepest_decile_x0depth_vs_x0'] = {
            'x0_mean_mm': float(b.mean()), 'x0depth_mean_mm': float(a.mean()),
            'mean_diff_mm': float(d_.mean()),
            'pct_change': float(100.0 * d_.mean() / b.mean()),
            'n_meshes': len(ok),
            'wins_for_x0depth': int((d_ < 0).sum()),
            't': float(t), 'p_ttest': float(p_t), 'p_wilcoxon': p_w,
            'cohens_dz': float(d_.mean() / d_.std(ddof=1)) if d_.std(ddof=1) > 0 else float('nan'),
        }
        r = tests['deepest_decile_x0depth_vs_x0']
        print("\nPAIRED TEST -- deepest decile, x0+depth vs x0 "
              f"(n={r['n_meshes']} meshes)")
        print(f"  x0        : {r['x0_mean_mm']:.3f} mm")
        print(f"  x0+depth  : {r['x0depth_mean_mm']:.3f} mm")
        print(f"  diff      : {r['mean_diff_mm']:+.3f} mm ({r['pct_change']:+.2f}%), "
              f"x0+depth better on {r['wins_for_x0depth']}/{r['n_meshes']} meshes")
        print(f"  paired t  : t={r['t']:.3f}  p={r['p_ttest']:.4f}   "
              f"Wilcoxon p={r['p_wilcoxon']:.4f}   dz={r['cohens_dz']:+.3f}")

    with open(os.path.join(args.out_dir, 'depth_deciles_summary.json'), 'w') as f:
        json.dump({'summary': summary, 'tests': tests, 'K': args.K,
                   'steps': steps, 'seed': args.seed, 'frac': args.frac,
                   'n_meshes': len(samples)}, f, indent=1)
    print(f"\nwrote {csv_path}")


if __name__ == '__main__':
    main()
