"""
C2 - Correspondence-native uncertainty benchmark.

The point of this script is NOT to invent a new uncertainty formula (rho, AUSE
and sparsification curves are standard, cf. Ilg et al. ECCV'18). It is to show
that fixed topology / full correspondence lets us evaluate completion
uncertainty PER VERTEX and PER ANATOMICAL REGION -- something the standard
correspondence-free completion metrics (MMD / TMD / UHD) structurally cannot do,
because an unordered point set has no "same point" across two shapes.

Four blocks (see docs/paper/C2_C4_PLAN.md):

  B1  uncertainty BASELINES. The obvious reviewer attack is "your uncertainty is
      just distance from the hole boundary". So we measure that, plus three more
      model-free proxies, on exactly the same vertices. Our per-vertex diffusion
      std has to beat all of them or the contribution does not stand.

  B2  REGION-WISE calibration. Template vertices are partitioned into k regions
      (k-means on template coordinates -- deterministic, no manual labels), and
      calibration is reported per region. Only possible under correspondence.

  B3  the DoF LAW. Per-hole geometry stats (curvature, hop distance from the
      mesh core, geodesic radius, missing %) regressed against the resulting
      error, to back the claim that error is governed by the local degrees of
      freedom of the missing region rather than by how much was removed.

  B4  SET-LEVEL metrics (MMD / TMD / UHD) on the same samples, for contrast:
      they collapse the whole shape to one number and cannot localise.

Outputs (to --out_dir, default <test_dir>/uncertainty_benchmark):
  benchmark.json, baselines.csv, regions.csv, dof.csv, setlevel.csv,
  per_mesh.csv, tables.tex, and plots (baselines.png, regions.png, dof.png).

Example:
  python scripts/eval_uncertainty_benchmark.py \
         --config_file config/config_diffusion_dfaust.cfg \
         --K 20 --regions 12
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
from collections import deque
from glob import glob
from trimesh import load_mesh

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from utils.masking import _build_adjacency
from scripts.infer_diffusion import (
    parse_diffusion_section, build_denoiser, get_template_info, complete,
)
from scripts.eval_uncertainty_calibration import metrics_from, sparsification


# --------------------------------------------------------------------------- #
# template-level quantities (fixed topology -> computed ONCE)
# --------------------------------------------------------------------------- #
def vertex_curvature(verts, adj):
    """Uniform-Laplacian (umbrella) magnitude: ||v_i - mean(neighbours)||.
    Low = flat (torso, cheek), high = detail/joints (fingers, nose, ears)."""
    curv = np.zeros(len(verts), dtype=np.float64)
    for i in range(len(verts)):
        nb = adj[i]
        if nb:
            curv[i] = np.linalg.norm(verts[i] - verts[nb].mean(axis=0))
    return curv


def multi_source_hop(adj, sources, n_vertices):
    """BFS hop distance to the NEAREST source vertex. With sources = the known
    region this is the 'distance from the hole boundary' proxy; with sources =
    a single core vertex it is the extremity measure."""
    dist = np.full(n_vertices, -1, dtype=np.int32)
    q = deque()
    for s in np.atleast_1d(sources):
        dist[s] = 0
        q.append(int(s))
    while q:
        v = q.popleft()
        for j in adj[v]:
            if dist[j] < 0:
                dist[j] = dist[v] + 1
                q.append(j)
    if (dist < 0).any():                      # disconnected -> treat as farthest
        dist[dist < 0] = dist.max() + 1 if dist.max() >= 0 else 0
    return dist


def connected_components(adj, member_mask):
    """Label connected components of the sub-mesh induced by `member_mask`.
    Returns (labels, sizes) with labels[-1] == -1 for non-members."""
    n = len(member_mask)
    labels = np.full(n, -1, dtype=np.int32)
    sizes = []
    cur = 0
    for start in np.where(member_mask)[0]:
        if labels[start] >= 0:
            continue
        q = deque([int(start)])
        labels[start] = cur
        cnt = 0
        while q:
            v = q.popleft()
            cnt += 1
            for j in adj[v]:
                if member_mask[j] and labels[j] < 0:
                    labels[j] = cur
                    q.append(j)
        sizes.append(cnt)
        cur += 1
    return labels, np.array(sizes, dtype=np.int32)


def mask_strategy_of(test_dir, base):
    """Recover the masking strategy from the saved partial's filename, e.g.
    test_004_partial_plane_cut_r0.12.ply -> 'plane_cut'.

    This matters for B3: a test set that mixes scattered and contiguous masks
    will show a strong error/missing-% correlation that is really just a
    'which mask type is this' detector -- scattered holes have MORE vertices
    missing yet are ~11x easier, because every missing vertex still has known
    neighbours. The DoF claim has to be made WITHIN a fixed hole topology.
    """
    import re
    for f in glob(os.path.join(test_dir, f'{base}_partial_*.ply')):
        m = re.match(rf'{re.escape(base)}_partial_(.+?)_r[\d.]+\.ply',
                     os.path.basename(f))
        if m:
            return m.group(1)
    return 'unknown'


def core_vertex(verts):
    """Most 'central' vertex of the template = closest to the centroid.
    Used as the seed for the extremity (hop-distance) measure."""
    return int(np.argmin(np.linalg.norm(verts - verts.mean(axis=0), axis=1)))


def training_vertex_variance(config, n_vertices, max_meshes=800):
    """Per-vertex std of the training shapes = a purely STATIC shape prior.
    A strong model-free uncertainty proxy: 'the model should be unsure wherever
    the population varies'. Returns (N,) or None if unavailable."""
    path = os.path.join(config['dataset_dir'], 'processed_completion', 'training.pt')
    if not os.path.exists(path):
        return None
    try:
        data, slices = torch.load(path, weights_only=False)
        y = data.y if getattr(data, 'y', None) is not None else data.x
        n_total = y.shape[0] // n_vertices
        if n_total < 2:
            return None
        y = y[:n_total * n_vertices].reshape(n_total, n_vertices, 3)
        if n_total > max_meshes:                       # subsample for speed
            idx = np.linspace(0, n_total - 1, max_meshes).astype(int)
            y = y[idx]
        return y.float().std(dim=0).norm(dim=1).numpy().astype(np.float64)
    except Exception as ex:
        print(f'  (train-variance proxy unavailable: {ex})')
        return None


# --------------------------------------------------------------------------- #
# B4 - set-level correspondence-free metrics, for contrast
# --------------------------------------------------------------------------- #
def _chamfer(a, b, device):
    """Symmetric Chamfer distance between two (N,3) point sets."""
    A = torch.as_tensor(a, dtype=torch.float32, device=device)
    B = torch.as_tensor(b, dtype=torch.float32, device=device)
    d = torch.cdist(A, B)                       # (Na, Nb)
    return float(d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean())


def _hausdorff_uni(a, b, device):
    """Unidirectional Hausdorff distance a -> b (max over a of min dist to b)."""
    A = torch.as_tensor(a, dtype=torch.float32, device=device)
    B = torch.as_tensor(b, dtype=torch.float32, device=device)
    return float(torch.cdist(A, B).min(dim=1)[0].max())


def set_level_metrics(stack, gt, partial_known, device, max_pairs=20):
    """MMD (quality), TMD (diversity), UHD (fidelity) -- the standard multimodal
    completion metrics. Deliberately computed on the SAME samples as the
    per-vertex metrics so the paper can contrast what each protocol can see."""
    K = len(stack)
    mmd = min(_chamfer(s, gt, device) for s in stack)
    pairs, tmd = 0, 0.0
    for i in range(K):
        for j in range(i + 1, K):
            tmd += _chamfer(stack[i], stack[j], device)
            pairs += 1
            if pairs >= max_pairs:
                break
        if pairs >= max_pairs:
            break
    tmd = tmd / max(pairs, 1)
    uhd = float(np.mean([_hausdorff_uni(partial_known, s, device) for s in stack]))
    return {'mmd': mmd, 'tmd': tmd, 'uhd': uhd}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--K', type=int, default=20, help='samples per mesh')
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps override')
    ap.add_argument('--test_dir', default=None,
                    help='single test set; repeat runs for different regimes')
    ap.add_argument('--test_dirs', nargs='*', default=None,
                    help='several test sets at once -> feeds the B3 DoF analysis')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--regions', type=int, default=12, help='k for region k-means')
    ap.add_argument('--max_meshes', type=int, default=None)
    ap.add_argument('--no_setlevel', action='store_true',
                    help='skip MMD/TMD/UHD (the slow O(K^2) block)')
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  K={args.K}  steps={num_steps}  regions={args.regions}')

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule']).to(device)
    template_mesh, _ = get_template_info(config)
    t_verts = np.array(template_mesh.vertices, dtype=np.float64)
    t_faces = np.array(template_mesh.faces)
    N_t = len(t_verts)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device)
        clip_value = scale.get('clip_value', None)
    else:
        sigma, clip_value = scale.to(device), None

    # ---- template-level precomputation (fixed topology -> once) ----
    print('precomputing template quantities ...')
    adj = _build_adjacency(t_faces, N_t)
    curv = vertex_curvature(t_verts, adj)
    hop_core = multi_source_hop(adj, [core_vertex(t_verts)], N_t)
    trainvar = training_vertex_variance(config, N_t)

    from sklearn.cluster import KMeans
    region_id = KMeans(n_clusters=args.regions, n_init=10,
                       random_state=args.seed).fit_predict(t_verts)

    # ---- which test sets ----
    if args.test_dirs:
        test_dirs = args.test_dirs
    elif args.test_dir:
        test_dirs = [args.test_dir]
    else:
        test_dirs = [os.path.join(config['dataset_dir'], 'completion_test_samples')]
    out_dir = args.out_dir or os.path.join(test_dirs[0], 'uncertainty_benchmark')
    os.makedirs(out_dir, exist_ok=True)

    PROXIES = ['ours', 'boundary', 'holesize', 'curvature', 'trainvar', 'oracle']
    pooled = {p: [] for p in PROXIES}
    pooled_err = []
    pooled_region, pooled_regime = [], []
    per_mesh, dof_rows, setlevel_rows = [], [], []

    for test_dir in test_dirs:
        regime = os.path.basename(os.path.normpath(test_dir))
        gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
        if not gt_files:
            nested = glob(os.path.join(test_dir, '*', '*_gt_complete.ply'))
            hint = (f'\n  did you mean --test_dir '
                    f'{os.path.dirname(nested[0])} ?' if nested else '')
            raise SystemExit(
                f'no *_gt_complete.ply found in {test_dir}{hint}\n'
                f'  (COMA keeps its test meshes in a test_set/ subfolder; '
                f'DFAUST does not)')
        if args.max_meshes:
            gt_files = gt_files[:args.max_meshes]
        print(f'\n=== {regime}: {len(gt_files)} meshes ===')

        for gt_file in gt_files:
            base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
            mask_file = os.path.join(test_dir, f'{base}_mask.npy')
            if not os.path.exists(mask_file):
                continue
            mask = np.load(mask_file)
            gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
            if len(gt) != N_t or len(mask) != N_t or not mask.any():
                continue

            partial = gt.copy()
            partial[mask] = 0.0
            samples = complete(denoiser, diffusion, partial, mask, config, device,
                               sigma, clip_value=clip_value, num_steps=num_steps,
                               num_samples=args.K)
            stack = np.stack(samples, axis=0)                 # (K,N,3)
            mu = stack.mean(axis=0)
            u_ours = np.linalg.norm(stack.std(axis=0), axis=1)
            err = np.linalg.norm(mu - gt, axis=1)

            # ---- B1 proxies, evaluated on the missing vertices only ----
            d_bnd = multi_source_hop(adj, np.where(~mask)[0], N_t).astype(np.float64)
            comp_lab, comp_sz = connected_components(adj, mask)
            u_hole = np.zeros(N_t)
            valid = comp_lab >= 0
            u_hole[valid] = comp_sz[comp_lab[valid]]

            prox = {
                'ours':      u_ours[mask],
                'boundary':  d_bnd[mask],
                'holesize':  u_hole[mask],
                'curvature': curv[mask],
                'oracle':    err[mask],
            }
            prox['trainvar'] = (trainvar[mask] if trainvar is not None
                                else np.zeros(int(mask.sum())))
            em = err[mask]

            for p in PROXIES:
                pooled[p].append(prox[p])
            pooled_err.append(em)
            pooled_region.append(region_id[mask])
            pooled_regime.append(np.array([regime] * int(mask.sum())))

            pm, _ = metrics_from(prox['ours'], em)
            pm.update(mesh=base, regime=regime,
                      missing_pct=float(100 * mask.sum() / N_t))
            per_mesh.append(pm)

            # ---- B3 per-hole geometry -> DoF row ----
            dof_rows.append({
                'mesh': base, 'regime': regime,
                'strategy': mask_strategy_of(test_dir, base),
                'missing_pct': pm['missing_pct'],
                'n_components': int(len(comp_sz)),
                'largest_component': int(comp_sz.max()) if len(comp_sz) else 0,
                'mean_curvature': float(curv[mask].mean()),
                'mean_hop_core': float(hop_core[mask].mean()),
                'max_hop_core': float(hop_core[mask].max()),
                'geodesic_radius': float(d_bnd[mask].max()),
                'mean_error': float(em.mean()),
                'mean_uncertainty': float(prox['ours'].mean()),
            })

            # ---- B4 set-level ----
            if not args.no_setlevel:
                sl = set_level_metrics(stack, gt, gt[~mask], device)
                sl.update(mesh=base, regime=regime,
                          missing_pct=pm['missing_pct'], mean_error=float(em.mean()))
                setlevel_rows.append(sl)

            print(f'  {base:12s} miss={mask.sum():5d} '
                  f'rho={pm["spearman_rho"]:.3f} AUSE={pm["ause"]:.4f} '
                  f'err={em.mean():.5f}')

    # ======================================================================= #
    E = np.concatenate(pooled_err)
    REG = np.concatenate(pooled_region)
    RGM = np.concatenate(pooled_regime)
    P = {p: np.concatenate(pooled[p]) for p in PROXIES}

    # ---- B1 table ----
    print('\n' + '=' * 72)
    print('B1  UNCERTAINTY BASELINES (pooled over all missing vertices)')
    print(f'{"proxy":<12}{"spearman":>10}{"pearson":>10}{"AUSE":>10}{"AURG":>10}')
    baselines = {}
    degenerate = []
    for p in PROXIES:
        # A proxy can be constant over the missing vertices -- e.g. with
        # scattered masks every missing vertex touches a known one, so the
        # boundary distance is 1 everywhere. That is NOT a win for us: the
        # proxy simply could not be tested on this mask regime, and the paper
        # has to say so rather than quietly counting it as beaten.
        if np.allclose(P[p], P[p][0]):
            baselines[p] = {'spearman_rho': float('nan'), 'pearson_r': float('nan'),
                            'ause': float('nan'), 'aurg': float('nan'),
                            'degenerate': True}
            degenerate.append(p)
        else:
            m, _ = metrics_from(P[p], E)
            m['degenerate'] = False
            baselines[p] = m
        b = baselines[p]
        flag = '   <- CONSTANT, not comparable' if b['degenerate'] else ''
        print(f'{p:<12}{b["spearman_rho"]:>10.4f}{b["pearson_r"]:>10.4f}'
              f'{b["ause"]:>10.5f}{b["aurg"]:>10.5f}{flag}')

    rivals = ['boundary', 'holesize', 'curvature', 'trainvar']
    tested = [p for p in rivals if not baselines[p]['degenerate']]
    ours_wins = all(baselines['ours']['ause'] <= baselines[p]['ause'] for p in tested)
    print(f'\n  -> ours beats {len(tested)}/{len(rivals)} testable model-free '
          f'proxies on AUSE: {ours_wins}')
    if degenerate:
        print(f'  !! degenerate on this mask regime (constant, excluded): '
              f'{", ".join(degenerate)}')
        print(f'     -> re-run on a CONTIGUOUS-mask regime to test them properly; '
              f'do not claim a win over a proxy that was never comparable.')
    if not ours_wins:
        print('  !! the C2 claim does NOT hold as stated -- report this honestly.')

    with open(os.path.join(out_dir, 'baselines.csv'), 'w') as f:
        f.write('proxy,spearman_rho,pearson_r,ause,aurg\n')
        for p in PROXIES:
            b = baselines[p]
            f.write(f'{p},{b["spearman_rho"]:.4f},{b["pearson_r"]:.4f},'
                    f'{b["ause"]:.5f},{b["aurg"]:.5f}\n')

    # ---- B2 region table ----
    print('\n' + '=' * 72)
    print('B2  REGION-WISE CALIBRATION (impossible without correspondence)')
    print(f'{"region":>7}{"n":>9}{"mean_err":>11}{"mean_unc":>11}{"rho":>9}{"AUSE":>9}')
    regions = []
    for k in range(args.regions):
        m = REG == k
        if m.sum() < 50:
            continue
        uk, ek = P['ours'][m], E[m]
        rm, _ = metrics_from(uk, ek) if not np.allclose(uk, uk[0]) else ({}, None)
        row = {'region': int(k), 'n_vertices': int(m.sum()),
               'centroid': t_verts[region_id == k].mean(axis=0).round(4).tolist(),
               'mean_error': float(ek.mean()), 'mean_uncertainty': float(uk.mean()),
               'mean_curvature': float(curv[region_id == k].mean()),
               'mean_hop_core': float(hop_core[region_id == k].mean()),
               'spearman_rho': rm.get('spearman_rho', float('nan')),
               'ause': rm.get('ause', float('nan'))}
        regions.append(row)
        print(f'{k:>7}{row["n_vertices"]:>9}{row["mean_error"]:>11.5f}'
              f'{row["mean_uncertainty"]:>11.5f}{row["spearman_rho"]:>9.3f}'
              f'{row["ause"]:>9.4f}')
    regions_sorted = sorted(regions, key=lambda r: -r['mean_error'])
    if len(regions_sorted) >= 2:
        hi, lo = regions_sorted[0], regions_sorted[-1]
        ratio = hi['mean_error'] / max(lo['mean_error'], 1e-12)
        print(f'\n  hardest region {hi["region"]} err={hi["mean_error"]:.5f} vs '
              f'easiest {lo["region"]} err={lo["mean_error"]:.5f}  -> {ratio:.1f}x spread')

    with open(os.path.join(out_dir, 'regions.csv'), 'w') as f:
        f.write('region,n_vertices,cx,cy,cz,mean_error,mean_uncertainty,'
                'mean_curvature,mean_hop_core,spearman_rho,ause\n')
        for r in regions:
            c = r['centroid']
            f.write(f'{r["region"]},{r["n_vertices"]},{c[0]},{c[1]},{c[2]},'
                    f'{r["mean_error"]:.6f},{r["mean_uncertainty"]:.6f},'
                    f'{r["mean_curvature"]:.6f},{r["mean_hop_core"]:.2f},'
                    f'{r["spearman_rho"]:.4f},{r["ause"]:.5f}\n')

    # ---- B3 DoF law ----
    print('\n' + '=' * 72)
    print('B3  THE DoF LAW: what actually predicts completion error?')
    from scipy.stats import spearmanr
    dof_corr, dof_within = {}, {}
    if len(dof_rows) >= 4:
        errs = np.array([d['mean_error'] for d in dof_rows])
        for key in ['missing_pct', 'mean_curvature', 'mean_hop_core',
                    'max_hop_core', 'geodesic_radius', 'largest_component',
                    'n_components']:
            v = np.array([d[key] for d in dof_rows], dtype=np.float64)
            rho = float(spearmanr(v, errs)[0]) if not np.allclose(v, v[0]) else float('nan')
            dof_corr[key] = rho
            print(f'  spearman(error, {key:<18}) = {rho:+.3f}')
        if not np.isnan(dof_corr.get('missing_pct', float('nan'))):
            best = max((k for k in dof_corr if k != 'missing_pct'),
                       key=lambda k: abs(dof_corr[k]) if not np.isnan(dof_corr[k]) else -1)
            print(f'\n  amount removed (missing_pct) rho={dof_corr["missing_pct"]:+.3f}  vs  '
                  f'best DoF proxy {best} rho={dof_corr[best]:+.3f}')

        # --- control for mask type -------------------------------------- #
        # The pooled correlations above are CONFOUNDED whenever the test set
        # mixes hole topologies: scattered masks remove more vertices yet are
        # far easier, so they drag missing_pct negative. Report the matched
        # comparison (error at similar missing %) and the within-strategy
        # correlation, which is what the paper should actually claim.
        strategies = sorted({d.get('strategy', 'unknown') for d in dof_rows})
        if len(strategies) > 1:
            print(f'\n  --- controlling for hole topology ---')
            print(f'  {"strategy":<16}{"n":>4}{"miss%":>9}{"mean_err":>11}'
                  f'{"rho(err,miss%)":>16}')
            for s in strategies:
                sub = [d for d in dof_rows if d.get('strategy') == s]
                e = np.array([d['mean_error'] for d in sub])
                p = np.array([d['missing_pct'] for d in sub])
                if len(sub) >= 4 and not np.allclose(p, p[0]):
                    rho_in = float(spearmanr(p, e)[0])
                    rho_s = f'{rho_in:+.3f}'
                else:
                    dof_within[s] = None
                    rho_s = f'n<4'
                    print(f'  {s:<16}{len(sub):>4}{p.mean():>9.1f}{e.mean():>11.5f}'
                          f'{rho_s:>16}')
                    continue
                dof_within[s] = {'n': len(sub), 'mean_missing_pct': float(p.mean()),
                                 'mean_error': float(e.mean()),
                                 'spearman_error_vs_missing_pct': rho_in}
                print(f'  {s:<16}{len(sub):>4}{p.mean():>9.1f}{e.mean():>11.5f}'
                      f'{rho_s:>16}')
            errs_by_s = {s: np.mean([d['mean_error'] for d in dof_rows
                                     if d.get('strategy') == s]) for s in strategies}
            hi = max(errs_by_s, key=errs_by_s.get)
            lo = min(errs_by_s, key=errs_by_s.get)
            ratio = errs_by_s[hi] / max(errs_by_s[lo], 1e-12)
            print(f'\n  -> at comparable missing %, "{hi}" is {ratio:.1f}x harder '
                  f'than "{lo}".')
            print(f'     THIS is the DoF claim; the pooled rho above is partly a '
                  f'mask-type detector.')
    else:
        print('  (need >=4 meshes; run with more test sets via --test_dirs)')

    with open(os.path.join(out_dir, 'dof.csv'), 'w') as f:
        cols = ['mesh', 'regime', 'strategy', 'missing_pct', 'n_components', 'largest_component',
                'mean_curvature', 'mean_hop_core', 'max_hop_core', 'geodesic_radius',
                'mean_error', 'mean_uncertainty']
        f.write(','.join(cols) + '\n')
        for d in dof_rows:
            f.write(','.join(str(d[c]) for c in cols) + '\n')

    # ---- B4 set-level contrast ----
    setlevel_summary = {}
    if setlevel_rows:
        print('\n' + '=' * 72)
        print('B4  SET-LEVEL METRICS (what the field currently reports)')
        with open(os.path.join(out_dir, 'setlevel.csv'), 'w') as f:
            f.write('mesh,regime,missing_pct,mmd,tmd,uhd,mean_error\n')
            for s in setlevel_rows:
                f.write(f'{s["mesh"]},{s["regime"]},{s["missing_pct"]:.2f},'
                        f'{s["mmd"]:.6f},{s["tmd"]:.6f},{s["uhd"]:.6f},'
                        f'{s["mean_error"]:.6f}\n')
        for regime in sorted({s['regime'] for s in setlevel_rows}):
            rs = [s for s in setlevel_rows if s['regime'] == regime]
            setlevel_summary[regime] = {
                k: float(np.mean([r[k] for r in rs]))
                for k in ['mmd', 'tmd', 'uhd', 'mean_error']}
            v = setlevel_summary[regime]
            print(f'  {regime:<32} MMD={v["mmd"]:.5f} TMD={v["tmd"]:.5f} '
                  f'UHD={v["uhd"]:.5f} | per-vertex err={v["mean_error"]:.5f}')
        print('  -> one number per shape: no location, no per-region breakdown.')

    # ---- per-mesh + json ----
    with open(os.path.join(out_dir, 'per_mesh.csv'), 'w') as f:
        f.write('mesh,regime,missing_pct,pearson_r,spearman_rho,ause,aurg,'
                'mean_uncertainty,mean_error\n')
        for m in per_mesh:
            f.write(f'{m["mesh"]},{m["regime"]},{m["missing_pct"]:.2f},'
                    f'{m["pearson_r"]:.4f},{m["spearman_rho"]:.4f},{m["ause"]:.5f},'
                    f'{m["aurg"]:.5f},{m["mean_uncertainty"]:.6f},'
                    f'{m["mean_error"]:.6f}\n')

    payload = {
        'config': {'config_file': args.config_file, 'K': args.K, 'steps': num_steps,
                   'regions': args.regions, 'test_dirs': test_dirs,
                   'n_meshes': len(per_mesh), 'n_vertices_pooled': int(len(E))},
        'B1_baselines': baselines, 'B1_ours_beats_all_proxies': bool(ours_wins),
        'B2_regions': regions,
        'B3_dof_spearman_pooled_CONFOUNDED': dof_corr,
        'B3_dof_within_strategy': dof_within, 'B3_rows': dof_rows,
        'B4_setlevel': setlevel_summary,
        'per_mesh': per_mesh,
    }
    with open(os.path.join(out_dir, 'benchmark.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    # ---- LaTeX ----
    tex = ['% auto-generated by eval_uncertainty_benchmark.py',
           '\\begin{tabular}{lrrrr}', '\\toprule',
           'uncertainty proxy & $\\rho$ & $r$ & AUSE $\\downarrow$ & AURG $\\uparrow$ \\\\',
           '\\midrule']
    names = {'ours': 'per-vertex diffusion std (ours)', 'boundary': 'distance to hole boundary',
             'holesize': 'hole size', 'curvature': 'template curvature',
             'trainvar': 'training-set variance', 'oracle': 'oracle (actual error)'}
    for p in PROXIES:
        b = baselines[p]
        tex.append(f'{names[p]} & {b["spearman_rho"]:.3f} & {b["pearson_r"]:.3f} & '
                   f'{b["ause"]:.4f} & {b["aurg"]:.4f} \\\\')
        if p == 'trainvar':
            tex.append('\\midrule')
    tex += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'tables.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 3.2))
        for p, style in [('ours', '-'), ('boundary', '--'), ('holesize', '-.'),
                         ('curvature', ':'), ('trainvar', (0, (3, 1, 1, 1)))]:
            if np.allclose(P[p], P[p][0]):
                continue
            fr, s, o, _ = sparsification(P[p], E)
            plt.plot(fr, s, linestyle=style, lw=1.6,
                     label=f'{p} (AUSE={baselines[p]["ause"]:.3f})')
        fr, s, o, _ = sparsification(P['ours'], E)
        plt.plot(fr, o, color='k', lw=1.2, alpha=.6, label='oracle')
        plt.xlabel('fraction of vertices removed (most-uncertain first)')
        plt.ylabel('normalised mean error')
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'baselines.png'), dpi=200)
        plt.close()

        if regions:
            plt.figure(figsize=(5, 3.2))
            x = [r['mean_uncertainty'] for r in regions]
            y = [r['mean_error'] for r in regions]
            plt.scatter(x, y, s=[max(8, r['n_vertices'] / 40) for r in regions],
                        c='#6C4FB6', alpha=.75)
            for r in regions:
                plt.annotate(str(r['region']), (r['mean_uncertainty'], r['mean_error']),
                             fontsize=6)
            plt.xlabel('region mean uncertainty')
            plt.ylabel('region mean error')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'regions.png'), dpi=200)
            plt.close()

        if len(dof_rows) >= 4:
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            errs = [d['mean_error'] for d in dof_rows]
            axes[0].scatter([d['missing_pct'] for d in dof_rows], errs,
                            s=14, c='#999')
            axes[0].set_xlabel('% vertices missing')
            axes[0].set_ylabel('mean error')
            axes[0].set_title(f'amount removed '
                              f'(rho={dof_corr.get("missing_pct", float("nan")):+.2f})',
                              fontsize=8)
            axes[1].scatter([d['mean_hop_core'] for d in dof_rows], errs,
                            s=14, c='#2E8B6F')
            axes[1].set_xlabel('mean hop distance from mesh core (DoF proxy)')
            axes[1].set_title(f'where it was removed '
                              f'(rho={dof_corr.get("mean_hop_core", float("nan")):+.2f})',
                              fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'dof.png'), dpi=200)
            plt.close()
        print(f'\nwritten to: {out_dir}')
    except Exception as ex:
        print(f'  (plots skipped: {ex})')


if __name__ == '__main__':
    main()
