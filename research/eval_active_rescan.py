"""
C4 - Uncertainty-guided next-best-scan.

Turns the per-vertex uncertainty map from a figure into a DECISION. Setting: a
scan came back with holes, the model completed them, and you are allowed to
re-acquire a limited amount of surface. Where do you point the scanner?

The naive answer is "at the biggest hole". Our own DFAUST numbers say that is
wrong -- a large flat hole in the back is nearly free to fill, a small hole
around the fingers is not. The claim here is that the uncertainty map picks
better than any area-based heuristic at equal budget.

Everything is simulated: we hold the ground-truth mesh, so "re-scanning a patch"
= revealing those vertices' true coordinates and re-completing.

Loop, per mesh, per strategy:

    mask = initial hole
    for round in 0..R:
        samples = complete(K)  ->  mu, u
        record (acquired_fraction, error over the ORIGINAL missing region)
        pick a contiguous patch of B% of the initially-missing vertices
        reveal it (mask=False, coords=GT)

Strategies:
    uncertainty   highest per-vertex sample std        <- ours
    random        uniform seed among missing            floor
    largest_hole  deepest vertex of the biggest hole    area-greedy
    boundary      closest to the known region           strong naive
    oracle        highest actual error                  ceiling

Error is always measured over the ORIGINAL missing region (revealed vertices
contribute zero error, because they are ground truth now). That keeps the curves
comparable across strategies and rounds -- it answers "how good is my final
mesh", which is the question that matters.

Outputs (to --out_dir, default <test_dir>/active_rescan):
  rescan_curves.csv, rescan_summary.json, rescan.png, rescan.tex

Example:
  python research/eval_active_rescan.py \
         --config_file research/configs/config_diffusion_dfaust.cfg \
         --test_dir data/DFAUST_ds/completion_test_samples \
         --K 10 --rounds 5 --budget 0.10 --max_meshes 10
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
from research.eval_uncertainty_benchmark import (
    multi_source_hop, connected_components,
)

STRATEGIES = ['uncertainty', 'random', 'largest_hole', 'boundary', 'oracle']


# --------------------------------------------------------------------------- #
def grow_patch(adj, missing, seed, size, rng):
    """BFS-grow a connected patch of `size` vertices through STILL-MISSING
    vertices, starting at `seed`. A real scanner re-acquires a contiguous
    region, not scattered vertices."""
    picked = {int(seed)}
    q = deque([int(seed)])
    while len(picked) < size and q:
        v = q.popleft()
        nb = [j for j in adj[v] if missing[j] and j not in picked]
        rng.shuffle(nb)
        for j in nb:
            picked.add(j)
            q.append(j)
            if len(picked) >= size:
                break
    # Patch exhausted before reaching `size` (hole smaller than the budget):
    # jump to another missing component and keep going.
    if len(picked) < size:
        rest = [i for i in np.where(missing)[0] if i not in picked]
        rng.shuffle(rest)
        for i in rest[:size - len(picked)]:
            picked.add(int(i))
    return np.array(sorted(picked), dtype=np.int64)


def pick_seed(strategy, missing, adj, unc, err, rng, n_vertices):
    """Choose the seed vertex for this round's patch, among missing vertices."""
    idx = np.where(missing)[0]
    if len(idx) == 0:
        return None
    if strategy == 'uncertainty':
        return int(idx[np.argmax(unc[idx])])
    if strategy == 'oracle':
        return int(idx[np.argmax(err[idx])])
    if strategy == 'random':
        return int(rng.choice(idx))
    if strategy == 'boundary':
        # closest to the known region = the safest, most "scannable" edge
        d = multi_source_hop(adj, np.where(~missing)[0], n_vertices)
        return int(idx[np.argmin(d[idx])])
    if strategy == 'largest_hole':
        labels, sizes = connected_components(adj, missing)
        if len(sizes) == 0:
            return int(idx[0])
        big = int(np.argmax(sizes))
        members = np.where(labels == big)[0]
        # deepest vertex of the biggest hole (furthest from any known vertex)
        d = multi_source_hop(adj, np.where(~missing)[0], n_vertices)
        return int(members[np.argmax(d[members])])
    raise ValueError(strategy)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--test_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--K', type=int, default=10, help='samples per completion')
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps override')
    ap.add_argument('--rounds', type=int, default=5, help='acquisition rounds')
    ap.add_argument('--budget', type=float, default=0.10,
                    help='fraction of the INITIAL hole re-acquired per round')
    ap.add_argument('--strategies', nargs='*', default=STRATEGIES)
    ap.add_argument('--scatter', action='store_true',
                    help='ablation: reveal scattered top-scoring vertices '
                         'instead of a contiguous patch')
    ap.add_argument('--target', type=float, default=0.5,
                    help='target error as a fraction of the round-0 error, for '
                         'the "budget to reach X" headline number')
    ap.add_argument('--max_meshes', type=int, default=10)
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule']).to(device)
    template_mesh, _ = get_template_info(config)
    t_faces = np.array(template_mesh.faces)
    N_t = len(template_mesh.vertices)
    adj = _build_adjacency(t_faces, N_t)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device)
        clip_value = scale.get('clip_value', None)
    else:
        sigma, clip_value = scale.to(device), None

    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples')
    out_dir = args.out_dir or os.path.join(test_dir, 'active_rescan')
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    if not gt_files:
        nested = glob(os.path.join(test_dir, '*', '*_gt_complete.ply'))
        hint = (f'\n  did you mean --test_dir {os.path.dirname(nested[0])} ?'
                if nested else '')
        raise SystemExit(
            f'no *_gt_complete.ply found in {test_dir}{hint}\n'
            f'  (COMA keeps its test meshes in a test_set/ subfolder; '
            f'DFAUST does not)')
    if args.max_meshes:
        gt_files = gt_files[:args.max_meshes]

    n_calls = len(gt_files) * len(args.strategies) * (args.rounds + 1)
    print(f'device={device}  K={args.K}  steps={num_steps}  rounds={args.rounds}  '
          f'budget={args.budget:.0%}/round')
    print(f'{len(gt_files)} meshes x {len(args.strategies)} strategies '
          f'-> {n_calls} completions x K={args.K} samples\n')

    rows = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(test_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        mask0 = np.load(mask_file)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t or len(mask0) != N_t or not mask0.any():
            continue
        n_missing0 = int(mask0.sum())
        per_round = max(1, int(round(args.budget * n_missing0)))
        print(f'{base}: {n_missing0} missing ({100*n_missing0/N_t:.1f}%), '
              f'{per_round} vertices/round')

        for strat in args.strategies:
            rng = np.random.RandomState(args.seed)      # same draw for all strategies
            mask = mask0.copy()
            partial = gt.copy()
            partial[mask] = 0.0

            for rnd in range(args.rounds + 1):
                acquired = int(mask0.sum() - mask.sum())
                # Same sampler randomness for every strategy at the same
                # (mesh, round). Without this, strategies get different
                # sampling luck and round 0 -- where no strategy has acquired
                # anything yet -- does not even start from a common error.
                torch.manual_seed(args.seed * 100003 + rnd)
                if mask.any():
                    samples = complete(denoiser, diffusion, partial, mask, config,
                                       device, sigma, clip_value=clip_value,
                                       num_steps=num_steps, num_samples=args.K)
                    stack = np.stack(samples, axis=0)
                    mu = stack.mean(axis=0)
                    unc = np.linalg.norm(stack.std(axis=0), axis=1)
                else:
                    mu, unc = gt.copy(), np.zeros(N_t)

                # Error over the ORIGINAL hole; revealed vertices are GT -> 0.
                full_err = np.linalg.norm(mu - gt, axis=1)
                full_err[~mask] = 0.0
                err_orig = float(full_err[mask0].mean())
                err_still = float(full_err[mask].mean()) if mask.any() else 0.0

                rows.append({'mesh': base, 'strategy': strat, 'round': rnd,
                             'acquired': acquired,
                             'acquired_frac': acquired / n_missing0,
                             'err_orig_missing': err_orig,
                             'err_still_missing': err_still,
                             'mean_uncertainty': float(unc[mask].mean())
                             if mask.any() else 0.0,
                             'n_missing': int(mask.sum())})

                if rnd == args.rounds or not mask.any():
                    break

                seed = pick_seed(strat, mask, adj, unc, full_err, rng, N_t)
                if seed is None:
                    break
                if args.scatter:
                    idx = np.where(mask)[0]
                    score = (unc if strat == 'uncertainty' else
                             full_err if strat == 'oracle' else
                             rng.rand(N_t))
                    pick = idx[np.argsort(-score[idx])[:per_round]]
                else:
                    pick = grow_patch(adj, mask, seed,
                                      min(per_round, int(mask.sum())), rng)
                mask[pick] = False
                partial[pick] = gt[pick]

            last = rows[-1]
            print(f'   {strat:<13} err {rows[-(last["round"]+1)]["err_orig_missing"]:.5f}'
                  f' -> {last["err_orig_missing"]:.5f}')

    # ======================================================================= #
    with open(os.path.join(out_dir, 'rescan_curves.csv'), 'w') as f:
        cols = ['mesh', 'strategy', 'round', 'acquired', 'acquired_frac',
                'err_orig_missing', 'err_still_missing', 'mean_uncertainty',
                'n_missing']
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    # ---- mean curve per strategy ----
    summary = {}
    print('\n' + '=' * 72)
    print('MEAN ERROR OVER THE ORIGINAL HOLE vs ACQUISITION BUDGET')
    hdr = 'strategy'.ljust(14) + ''.join(
        f'{r*args.budget:>9.0%}' for r in range(args.rounds + 1))
    print(hdr + '     AUC')
    curves = {}
    for strat in args.strategies:
        curve = []
        for rnd in range(args.rounds + 1):
            v = [r['err_orig_missing'] for r in rows
                 if r['strategy'] == strat and r['round'] == rnd]
            curve.append(float(np.mean(v)) if v else float('nan'))
        curves[strat] = curve
        fr = np.arange(args.rounds + 1) * args.budget
        auc = float(np.trapz(curve, fr)) if len(curve) > 1 else float('nan')
        summary[strat] = {'curve': curve, 'budget_fracs': fr.tolist(), 'auc': auc}
        print(strat.ljust(14) + ''.join(f'{c:>9.5f}' for c in curve) + f'  {auc:>8.5f}')

    # ---- headline: budget needed to reach `target` x round-0 error ----
    base_err = np.mean([curves[s][0] for s in args.strategies])
    thresh = args.target * base_err
    print(f'\nBudget to reach {args.target:.0%} of the initial error '
          f'({thresh:.5f}):')
    budgets = {}
    for strat in args.strategies:
        c = curves[strat]
        hit = next((i for i, v in enumerate(c) if v <= thresh), None)
        if hit is None:
            budgets[strat] = None
            print(f'  {strat:<14} not reached within {args.rounds*args.budget:.0%}')
        else:
            # linear interpolation between the bracketing rounds
            if hit == 0:
                b = 0.0
            else:
                y0, y1 = c[hit - 1], c[hit]
                t = (y0 - thresh) / max(y0 - y1, 1e-12)
                b = (hit - 1 + t) * args.budget
            budgets[strat] = float(b)
            print(f'  {strat:<14} {b:.1%} of the hole')
    if budgets.get('uncertainty') is not None:
        for rival in ['largest_hole', 'boundary', 'random']:
            if budgets.get(rival):
                save = 1 - budgets['uncertainty'] / budgets[rival]
                word = 'less' if save > 0 else 'MORE'
                print(f'  -> ours needs {abs(save):.0%} {word} budget than {rival}')

    with open(os.path.join(out_dir, 'rescan_summary.json'), 'w') as f:
        json.dump({'config': {'config_file': args.config_file, 'test_dir': test_dir,
                              'K': args.K, 'steps': num_steps, 'rounds': args.rounds,
                              'budget': args.budget, 'scatter': args.scatter,
                              'target': args.target, 'n_meshes': len(gt_files)},
                   'curves': summary, 'budget_to_target': budgets,
                   'target_error': thresh}, f, indent=2)

    # ---- LaTeX ----
    tex = ['% auto-generated by eval_active_rescan.py',
           '\\begin{tabular}{l' + 'r' * (args.rounds + 1) + 'r}', '\\toprule',
           'rescanning strategy & ' +
           ' & '.join(f'{r*args.budget:.0%}' for r in range(args.rounds + 1)) +
           ' & AUC \\\\', '\\midrule']
    label = {'uncertainty': 'uncertainty-guided (ours)', 'random': 'random',
             'largest_hole': 'largest hole (area-greedy)',
             'boundary': 'hole boundary', 'oracle': 'oracle (upper bound)'}
    for strat in args.strategies:
        tex.append(label.get(strat, strat) + ' & ' +
                   ' & '.join(f'{c:.4f}' for c in curves[strat]) +
                   f' & {summary[strat]["auc"]:.4f} \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'rescan.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        style = {'uncertainty': ('-', '#6C4FB6', 2.2), 'random': (':', '#999', 1.4),
                 'largest_hole': ('--', '#C2703D', 1.6),
                 'boundary': ('-.', '#2E8B6F', 1.6), 'oracle': ('-', '#000', 1.0)}
        plt.figure(figsize=(5, 3.4))
        fr = np.arange(args.rounds + 1) * args.budget * 100
        for strat in args.strategies:
            ls, c, lw = style.get(strat, ('-', None, 1.5))
            plt.plot(fr, curves[strat], linestyle=ls, color=c, lw=lw,
                     marker='o', ms=3, label=label.get(strat, strat))
        plt.axhline(thresh, color='r', lw=.8, alpha=.5,
                    label=f'{args.target:.0%} of initial error')
        plt.xlabel('% of the original hole re-acquired')
        plt.ylabel('mean error over the original hole')
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'rescan.png'), dpi=200)
        plt.close()
        print(f'\nwritten to: {out_dir}')
    except Exception as ex:
        print(f'  (plot skipped: {ex})')


if __name__ == '__main__':
    main()
