"""
Quantitative evaluation of the per-vertex uncertainty produced by drawing K
diffusion completions (the paper's key novelty, turned from qualitative into a
number).

For each test mesh we draw K completions with the SAME sampler used everywhere
(scripts.infer_diffusion.complete), then per missing vertex i:
    prediction  mu_i  = mean over K samples            (point estimate)
    uncertainty u_i   = ||std over K samples||_2        (per-vertex std magnitude)
    error       e_i   = ||mu_i - gt_i||_2               (actual error)
Known vertices are exact (u=e=0 by construction), so calibration is measured on
the MISSING region only.

Reported (pooled over all missing vertices of all test meshes):
  * Pearson r and Spearman rho between u and e   -> does uncertainty track error?
  * Sparsification curve + AUSE (Area Under the Sparsification Error), the
    standard uncertainty metric (Ilg et al., ECCV'18): remove the most-uncertain
    vertices first and watch mean error fall; compare to the oracle (remove
    highest-error first). Lower AUSE = better-calibrated. AURG (area under the
    random gain) is also reported (higher = better).
  * A reliability table (mean error per uncertainty decile; should increase).

Outputs (to --out_dir, default data/completion_test_samples/calibration):
  calibration_metrics.json, calibration_metrics.csv, per_mesh.csv,
  sparsification.png, reliability.png, scatter.png, and a LaTeX snippet.

Run (on the GPU box that has the trained diffusion checkpoint):
  python research/eval_uncertainty_calibration.py \
         --config_file research/configs/config_diffusion.cfg --K 20
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import configparser
import numpy as np
import torch
from glob import glob
from trimesh import load_mesh

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (
    parse_diffusion_section, build_denoiser, get_template_info, complete,
)


# --------------------------------------------------------------------------- #
def sparsification(u, e, n_frac=100):
    """Return fractions t, sparsification s(t), oracle o(t), random baseline r(t).
    All curves normalised so value at t=0 is 1 (mean error over all vertices)."""
    n = len(e)
    base = e.mean()
    if base <= 0:
        base = 1.0
    fracs = np.linspace(0.0, 1.0, n_frac, endpoint=False)
    order_u = np.argsort(-u)          # most-uncertain first
    order_e = np.argsort(-e)          # highest-error first (oracle)
    e_u = e[order_u]
    e_e = e[order_e]
    s = np.empty(n_frac)
    o = np.empty(n_frac)
    for j, t in enumerate(fracs):
        k = int(t * n)
        rem = n - k
        s[j] = e_u[k:].mean() / base if rem > 0 else 0.0
        o[j] = e_e[k:].mean() / base if rem > 0 else 0.0
    r = np.ones(n_frac)               # removing random vertices leaves mean ~unchanged
    return fracs, s, o, r


def metrics_from(u, e):
    from scipy.stats import pearsonr, spearmanr
    out = {}
    out['n_vertices'] = int(len(e))
    out['pearson_r'] = float(pearsonr(u, e)[0])
    out['spearman_rho'] = float(spearmanr(u, e)[0])
    fracs, s, o, r = sparsification(u, e)
    out['ause'] = float(np.trapz(s - o, fracs))          # lower is better
    out['aurg'] = float(np.trapz(r - s, fracs))          # higher is better
    out['mean_uncertainty'] = float(u.mean())
    out['mean_error'] = float(e.mean())
    # reliability: mean error per uncertainty decile
    q = np.quantile(u, np.linspace(0, 1, 11))
    rel = []
    for b in range(10):
        m = (u >= q[b]) & (u <= q[b + 1] if b == 9 else u < q[b + 1])
        rel.append(float(e[m].mean()) if m.any() else float('nan'))
    out['reliability_deciles'] = rel
    return out, (fracs, s, o, r)


# --------------------------------------------------------------------------- #
# Calibration curve: is the uncertainty on the same SCALE as the error, not
# merely correlated with it? Correlation/AUSE only test ranking -- multiply every
# u by 10 and both are unchanged. This asks the question a user actually has:
# "the model says u; how many mm of error should I expect?"
# --------------------------------------------------------------------------- #
def calibration_curve(u, e, n_bins=15):
    """Equal-COUNT bins over u; per bin the mean u, mean e and the s.e.m. of e.

    Equal-count rather than equal-width: u is heavily right-skewed, so equal-width
    bins put almost every vertex in the first bin and leave the high-u bins with a
    handful of points each and error bars too wide to read.
    """
    edges = np.quantile(u, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)     # make the last bin closed
    idx = np.clip(np.digitize(u, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 2:
            continue
        eb = e[m]
        rows.append({'bin': b, 'n': int(m.sum()),
                     'mean_u': float(u[m].mean()),
                     'mean_e': float(eb.mean()),
                     'sem_e': float(eb.std(ddof=1) / np.sqrt(m.sum()))})
    return rows


def _analytic_slope(K):
    """Closed-form approximation to the perfectly-calibrated slope, for this K.

    Not 1.0, and worth stating explicitly before anyone reads a slope of 0.92 as
    miscalibration. Assume the posterior at a vertex is isotropic Gaussian with
    per-axis sigma, and that the ground truth is a draw from it. Then:

      * u = ||per-axis std over K samples||_2. Each component is a sample std
        computed with ddof=0 (numpy's default, which is what the sampler above
        uses), so E[component] = sqrt((K-1)/K) * c4(K) * sigma -- BELOW sigma on
        two counts.
      * e = ||mean of K samples - truth||. That difference has per-axis variance
        sigma^2 (1 + 1/K), and its norm is chi-distributed with 3 dof, so
        E[e] = 2 sqrt(2/pi) * sigma * sqrt(1 + 1/K) -- ABOVE sigma.

    The two finite-K corrections push in opposite directions and largely cancel
    (K=20 gives 0.981). This is used only to set the SCALE of the bootstrap in
    ideal_slope(); it is not the reference line, because it is biased ~3% high:
    it approximates E||s|| by ||E s|| (the norm is convex, so the truth is
    larger, which makes the true slope smaller), and it cannot see the
    regression-to-the-mean induced by binning on a noisy u.
    """
    from scipy.special import gammaln
    c4 = np.sqrt(2.0 / (K - 1)) * np.exp(gammaln(K / 2.0) - gammaln((K - 1) / 2.0))
    e_u = np.sqrt(3.0) * np.sqrt((K - 1) / K) * c4          # E[u] / sigma
    e_e = 2.0 * np.sqrt(2.0 / np.pi) * np.sqrt(1.0 + 1.0 / K)   # E[e] / sigma
    return float(e_e / e_u)


def ideal_slope(u_obs, K, n_rep=5, seed=0, n_bins=15):
    """Reference slope, by parametric bootstrap: what slope would this exact
    pipeline report if the model were PERFECTLY calibrated?

    Rather than trusting a closed form, synthesise a posterior that is calibrated
    by construction -- per-vertex sigma taken from the observed u, K samples drawn
    from it, and the "truth" drawn from the same distribution -- then push it
    through the same binning and the same weighted fit. Every artifact of the
    estimator (ddof=0 sample std, finite K, binning on a noisy u) is then present
    in the reference on equal terms, so the comparison isolates real
    miscalibration instead of measuring the estimator's own bias.
    """
    rng = np.random.RandomState(seed)
    sigma = np.asarray(u_obs, dtype=np.float64) / max(_analytic_slope(K), 1e-12)
    sigma = sigma / np.sqrt(3.0)                      # per-axis, approximately
    n = len(sigma)
    slopes = []
    for _ in range(n_rep):
        S = rng.randn(K, n, 3) * sigma[None, :, None]
        gt = rng.randn(n, 3) * sigma[:, None]
        u = np.linalg.norm(S.std(axis=0), axis=1)
        e = np.linalg.norm(S.mean(axis=0) - gt, axis=1)
        rows = calibration_curve(u, e, n_bins=n_bins)
        mu = np.array([r['mean_u'] for r in rows])
        me = np.array([r['mean_e'] for r in rows])
        w = np.array([r['n'] for r in rows], dtype=float)
        A = np.vstack([mu, np.ones_like(mu)]).T
        W = np.sqrt(w)[:, None]
        (sl, _b), *_ = np.linalg.lstsq(A * W, me * W[:, 0], rcond=None)
        slopes.append(float(sl))
    return float(np.mean(slopes)), float(np.std(slopes))


def calibration_fit(rows, u, e, K):
    """Weighted line fit through the binned curve + scale-calibration metrics."""
    mu = np.array([r['mean_u'] for r in rows])
    me = np.array([r['mean_e'] for r in rows])
    w = np.array([r['n'] for r in rows], dtype=float)

    # Weight by bin population: every bin holds the same count by construction,
    # but a bin dropped for being under-populated should not silently reweight
    # the rest.
    A = np.vstack([mu, np.ones_like(mu)]).T
    W = np.sqrt(w)[:, None]
    (slope, intercept), *_ = np.linalg.lstsq(A * W, me * W[:, 0], rcond=None)

    pred = slope * mu + intercept
    ss_res = float((w * (me - pred) ** 2).sum())
    ss_tot = float((w * (me - np.average(me, weights=w)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # ENCE (Levi et al.): mean relative gap between predicted and observed error
    # across bins. 0 = perfectly calibrated scale.
    ence = float(np.mean(np.abs(mu - me) / np.maximum(mu, 1e-12)))

    # The single scalar that best maps u -> e, i.e. argmin_s sum (e - s u)^2.
    # This is what you would multiply the uncertainty map by to read it in mm.
    s_opt = float((u * e).sum() / max((u * u).sum(), 1e-24))

    ideal, ideal_sd = ideal_slope(u, K, n_bins=len(rows))
    return {'slope': float(slope), 'intercept': float(intercept), 'r2': float(r2),
            'ence': ence, 'scale_opt': s_opt,
            'ideal_slope': ideal, 'ideal_slope_sd': ideal_sd,
            'analytic_slope': _analytic_slope(K),
            'slope_over_ideal': float(slope / ideal),
            'mean_ratio': float(e.mean() / max(u.mean(), 1e-24)),
            'n_bins_used': len(rows)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--param', default='eps', choices=['eps', 'x0', 'v'],
                    help="prediction target of the arm being scored. MUST match how the "
                         "arm was trained -- sampling an x0 arm as eps produces garbage.")
    ap.add_argument('--tag', default=None,
                    help="checkpoint subfolder suffix -> predict_<param>[_<tag>]/")
    ap.add_argument('--K', type=int, default=20, help='samples per mesh')
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps override')
    ap.add_argument('--test_dir', default=None)
    ap.add_argument('--gt_dir', default=None, help="split layout: <test_dir>/gt")
    ap.add_argument('--mask_dir', default=None, help="split layout: <test_dir>/partial")
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--max_meshes', type=int, default=None, help='limit for a quick run')
    ap.add_argument('--bins', type=int, default=15,
                    help='equal-count bins for the calibration curve (u vs e)')
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--use_last', action='store_true',
                    help="load diffusion_denoiser_ema_last.pt (LAST epoch) instead of "
                         "diffusion_denoiser_ema.pt (BEST-EVAL epoch); epoch-matching")
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

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema,
                              use_last=args.use_last)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=args.param).to(device)
    template_mesh, _ = get_template_info(config)
    N_t = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma = scale['sigma'].to(device); clip_value = scale.get('clip_value', None)
    else:
        sigma = scale.to(device); clip_value = None

    test_dir = args.test_dir or os.path.join(config['dataset_dir'], 'completion_test_samples')
    gt_dir = args.gt_dir or test_dir
    mask_dir = args.mask_dir or test_dir
    out_dir = args.out_dir or os.path.join(test_dir, 'calibration')
    os.makedirs(out_dir, exist_ok=True)
    gt_files = sorted(glob(os.path.join(gt_dir, '*_gt_complete.ply')))
    if args.max_meshes:
        gt_files = gt_files[:args.max_meshes]
    print(f'{len(gt_files)} test meshes\n')

    all_u, all_e = [], []
    per_mesh = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(mask_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        mask = np.load(mask_file)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t or len(mask) != N_t or not mask.any():
            continue
        partial = gt.copy(); partial[mask] = 0.0

        samples = complete(denoiser, diffusion, partial, mask, config, device,
                           sigma, clip_value=clip_value, num_steps=num_steps,
                           num_samples=args.K)
        stack = np.stack(samples, axis=0)                     # (K,N,3)
        mu = stack.mean(axis=0)
        u = np.linalg.norm(stack.std(axis=0), axis=1)         # (N,)
        e = np.linalg.norm(mu - gt, axis=1)                   # (N,)
        um, em = u[mask], e[mask]
        all_u.append(um); all_e.append(em)

        pm, _ = metrics_from(um, em)
        pm['mesh'] = base; pm['missing_pct'] = float(100 * mask.sum() / len(mask))
        per_mesh.append(pm)
        print(f'  {base}: miss={mask.sum():4d}  r={pm["pearson_r"]:.3f}  '
              f'rho={pm["spearman_rho"]:.3f}  AUSE={pm["ause"]:.4f}')

    U = np.concatenate(all_u); E = np.concatenate(all_e)
    agg, curves = metrics_from(U, E)
    agg['K'] = args.K; agg['steps'] = num_steps; agg['n_meshes'] = len(per_mesh)
    agg['per_mesh_ause_mean'] = float(np.mean([m['ause'] for m in per_mesh]))
    agg['per_mesh_ause_std'] = float(np.std([m['ause'] for m in per_mesh]))
    agg['per_mesh_pearson_mean'] = float(np.mean([m['pearson_r'] for m in per_mesh]))
    agg['per_mesh_spearman_mean'] = float(np.mean([m['spearman_rho'] for m in per_mesh]))

    # ---- calibration curve: does u map onto e on the right SCALE? ----
    curve_rows = calibration_curve(U, E, n_bins=args.bins)
    fit = calibration_fit(curve_rows, U, E, args.K)
    agg['calibration_curve'] = curve_rows
    agg['calibration_fit'] = fit

    # Keep the pooled per-vertex arrays so the curve can be re-binned or re-plotted
    # without paying for another K-sample pass over the test set.
    np.savez_compressed(os.path.join(out_dir, 'pooled_u_e.npz'), u=U, e=E)

    # ---- save numbers ----
    with open(os.path.join(out_dir, 'calibration_metrics.json'), 'w') as f:
        json.dump({'aggregate': agg, 'per_mesh': per_mesh}, f, indent=2)
    with open(os.path.join(out_dir, 'per_mesh.csv'), 'w') as f:
        f.write('mesh,missing_pct,pearson_r,spearman_rho,ause,aurg,mean_uncertainty,mean_error\n')
        for m in per_mesh:
            f.write(f'{m["mesh"]},{m["missing_pct"]:.2f},{m["pearson_r"]:.4f},'
                    f'{m["spearman_rho"]:.4f},{m["ause"]:.5f},{m["aurg"]:.5f},'
                    f'{m["mean_uncertainty"]:.6f},{m["mean_error"]:.6f}\n')

    print('\n' + '=' * 60)
    print('AGGREGATE (pooled over missing vertices of all meshes)')
    print(f'  vertices           : {agg["n_vertices"]}')
    print(f'  Pearson r          : {agg["pearson_r"]:.4f}')
    print(f'  Spearman rho       : {agg["spearman_rho"]:.4f}')
    print(f'  AUSE (lower better): {agg["ause"]:.5f}')
    print(f'  AURG (higher better): {agg["aurg"]:.5f}')
    print(f'  per-mesh Pearson   : {agg["per_mesh_pearson_mean"]:.4f}')
    print(f'  per-mesh AUSE      : {agg["per_mesh_ause_mean"]:.5f} +/- {agg["per_mesh_ause_std"]:.5f}')

    print('\nCALIBRATION CURVE (mean error vs mean uncertainty, '
          f'{fit["n_bins_used"]} equal-count bins)')
    print(f'  {"bin":>4s} {"n":>7s} {"mean u (mm)":>12s} {"mean e (mm)":>12s} {"e/u":>7s}')
    for r in curve_rows:
        print(f'  {r["bin"]:>4d} {r["n"]:>7d} {1000*r["mean_u"]:>12.4f} '
              f'{1000*r["mean_e"]:>12.4f} {r["mean_e"]/max(r["mean_u"],1e-12):>7.3f}')
    print(f'\n  fit: e = {fit["slope"]:.4f} * u + {fit["intercept"]*1000:+.4f} mm'
          f'   (R^2 = {fit["r2"]:.4f})')
    print(f'  ideal slope for K={args.K} (perfect calibration): {fit["ideal_slope"]:.4f}')
    print(f'  slope / ideal      : {fit["slope_over_ideal"]:.4f}  '
          f'({"under" if fit["slope_over_ideal"] > 1 else "over"}-confident)')
    print(f'  ENCE (0 = perfect) : {fit["ence"]:.4f}')
    print(f'  best scalar u->e   : {fit["scale_opt"]:.4f}')
    print(f'  mean e / mean u    : {fit["mean_ratio"]:.4f}')

    # ---- LaTeX snippet ----
    tex = (f"% auto-generated by eval_uncertainty_calibration.py\n"
           f"Pearson $r={agg['pearson_r']:.3f}$, Spearman "
           f"$\\rho={agg['spearman_rho']:.3f}$, AUSE $={agg['ause']:.4f}$ "
           f"(AURG $={agg['aurg']:.4f}$), over {agg['n_meshes']} meshes, $K={args.K}$.\n"
           f"Calibration curve: $e = {fit['slope']:.3f}\\,u "
           f"{fit['intercept']*1000:+.4f}$\\,mm, $R^2={fit['r2']:.3f}$; ideal slope "
           f"for $K={args.K}$ is ${fit['ideal_slope']:.3f}$ "
           f"(ratio ${fit['slope_over_ideal']:.3f}$), ENCE $={fit['ence']:.3f}$.\n")
    with open(os.path.join(out_dir, 'calibration_snippet.tex'), 'w') as f:
        f.write(tex)
    with open(os.path.join(out_dir, 'calibration_curve.csv'), 'w') as f:
        f.write('bin,n,mean_u_mm,mean_e_mm,sem_e_mm,ratio\n')
        for r in curve_rows:
            f.write(f'{r["bin"]},{r["n"]},{1000*r["mean_u"]:.6f},'
                    f'{1000*r["mean_e"]:.6f},{1000*r["sem_e"]:.6f},'
                    f'{r["mean_e"]/max(r["mean_u"],1e-12):.6f}\n')

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fracs, s, o, r = curves
        plt.figure(figsize=(4, 3))
        plt.plot(fracs, s, label=f'ours (AUSE={agg["ause"]:.3f})', lw=2)
        plt.plot(fracs, o, '--', label='oracle', lw=1.5, color='#444')
        plt.plot(fracs, r, ':', label='random', lw=1.2, color='#999')
        plt.xlabel('fraction of vertices removed (most-uncertain first)')
        plt.ylabel('normalised mean error')
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'sparsification.png'), dpi=200)
        plt.close()

        rel = agg['reliability_deciles']
        plt.figure(figsize=(4, 3))
        plt.bar(range(1, 11), rel, color='#6C4FB6')
        plt.xlabel('uncertainty decile (low -> high)')
        plt.ylabel('mean actual error')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'reliability.png'), dpi=200)
        plt.close()

        # Calibration curve: the plot Oliver asked for. Points are equal-count
        # bins; the dashed reference is the ideal slope, NOT y=x (see ideal_slope).
        mu = np.array([r['mean_u'] for r in curve_rows]) * 1000.0
        me = np.array([r['mean_e'] for r in curve_rows]) * 1000.0
        se = np.array([r['sem_e'] for r in curve_rows]) * 1000.0
        lim = float(max(mu.max(), me.max())) * 1.05
        plt.figure(figsize=(4, 3.6))
        xs = np.linspace(0, lim, 50)
        plt.plot(xs, fit['ideal_slope'] * xs, '--', color='#444', lw=1.3,
                 label=f'ideal (slope {fit["ideal_slope"]:.2f})')
        plt.plot(xs, fit['slope'] * xs + fit['intercept'] * 1000.0, '-',
                 color='#C1543A', lw=1.6,
                 label=f'fit: {fit["slope"]:.2f}u{fit["intercept"]*1000:+.3f}'
                       f'  ($R^2$={fit["r2"]:.3f})')
        plt.errorbar(mu, me, yerr=se, fmt='o', ms=4, color='#2E5FA3',
                     ecolor='#2E5FA3', elinewidth=1, capsize=2, label='binned')
        plt.xlim(0, lim); plt.ylim(0, lim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('mean predicted uncertainty (mm)')
        plt.ylabel('mean observed error (mm)')
        plt.legend(fontsize=7, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'calibration_curve.png'), dpi=200)
        plt.close()

        idx = np.random.choice(len(U), min(20000, len(U)), replace=False)
        plt.figure(figsize=(4, 3))
        plt.scatter(U[idx], E[idx], s=2, alpha=0.15, color='#2E8B6F')
        plt.xlabel('predicted uncertainty (per-vertex std)')
        plt.ylabel('actual error')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'scatter.png'), dpi=200)
        plt.close()
        print(f'  plots + metrics written to: {out_dir}')
    except Exception as ex:
        print(f'  (plots skipped: {ex})')


if __name__ == '__main__':
    main()
