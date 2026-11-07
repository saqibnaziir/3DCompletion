"""
Run the trained diffusion model on the legacy cropped partial meshes
(data/test_old/partial/no_*.ply).

These are REAL partials: vertices were physically removed, so they have fewer
than 5023 vertices and are NOT in template topology. The model needs the full
template topology, so we first remap each partial into it:

  - The matching complete mesh (data/test_old/gt/gt_*.ply) IS in template
    topology (5023 verts). The partial is an exact subset of it.
  - A template vertex is KNOWN if it appears in the partial, else MISSING.
  - Build the 5023-vertex partial (missing zeroed), run completion, compare to GT.

Outputs completed .ply files, a GT | Input | Ours | Error figure per sample, and
(when --num_samples > 1) the same single-draw vs mean-of-K metrics that
`diag_sampling_floor.py` reports, written to `metrics.csv` + `summary.txt`.

Usage:
  python research/complete_real_partial.py --param x0 --tag depthw_nrm_scratch \
      --partial_dir ... --gt_dir ... --out_dir ... --num_samples 10
"""
import os, sys, argparse, csv, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.sparse as sp
import torch
import pyvista as pv
from glob import glob
from trimesh import Trimesh, load_mesh
from scipy.spatial import cKDTree

from config.config import read_config, resolve_config_file
from utils.funcs import save_ply_explicit
from utils.diffusion import GaussianDiffusion
from utils.normals import normal_angular_error, boundary_band
from utils.boundary_field import BoundaryFieldSolver
from scripts.infer_diffusion import (build_denoiser, parse_diffusion_section,
                                     get_template_info, complete)
from research.qualitative_figure import add_surface, pv_faces, bold_text


def uniform_laplacian(adj, n):
    """L = I - D^-1 A (umbrella). Used only for the roughness ratio."""
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sp.eye(n, format='csr') - sp.diags(1.0 / deg) @ adj


def mesh_stats(verts, gt, faces, mask, band, L):
    """Normal + position + roughness numbers for one candidate mesh.

    Same definitions as research/diag_sampling_floor.py so the numbers here are
    directly comparable to the COMA test-set tables.
    """
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
        'pos_p95_mm': float(1000.0 * np.percentile(pos[mask], 95)) if mask.any() else 0.0,
        'pos_max_mm': float(1000.0 * pos[mask].max()) if mask.any() else 0.0,
        'rough_ratio': float(lc[mask].mean() / max(lg[mask].mean(), 1e-12))
                       if mask.any() else 1.0,
    }


def gt_for(partial_name, gt_dir):
    """Map no_<key>.ply -> gt_<key>.ply (handles plane/plain naming)."""
    key = partial_name.replace('no_', '').replace('.ply', '')
    cand = os.path.join(gt_dir, f'gt_{key}.ply')
    if os.path.exists(cand):
        return cand
    if key == 'plane':
        alt = os.path.join(gt_dir, 'gt_plain.ply')
        if os.path.exists(alt):
            return alt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--param', default=None, choices=['eps', 'x0', 'v'],
                    help="prediction target; also selects the predict_<param> "
                         "checkpoint subfolder. Omit for the legacy base-dir eps model.")
    ap.add_argument('--tag', default=None,
                    help="checkpoint subfolder suffix, matching train_diffusion.py "
                         "--tag (e.g. --param x0 --tag depthw_nrm_scratch)")
    ap.add_argument('--use_last', action='store_true',
                    help="load diffusion_denoiser_ema_last.pt (LAST epoch) instead of "
                         "diffusion_denoiser_ema.pt (BEST-EVAL epoch)")
    ap.add_argument('--no_ema', action='store_true')
    ap.add_argument('--partial_dir', default=os.path.join('data', 'test_old', 'partial'))
    ap.add_argument('--gt_dir', default=os.path.join('data', 'test_old', 'gt'))
    ap.add_argument('--out_dir', default=os.path.join('data', 'test_old', 'completed_diffusion'))
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--clip', type=float, default=None,
                    help="sampling clamp override; only affects eps-prediction "
                         "(default: the value stored in diffusion_scale.pt)")
    ap.add_argument('--num_samples', type=int, default=1,
                    help="K draws per mesh. K>1 enables the single-draw vs mean-of-K "
                         "metrics and the diversity/uncertainty figures.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--unc_cmap', default='jet',
                    help="colormap for the uncertainty panel. Keep 'jet': every "
                         "other paper figure uses it, and a second colormap here "
                         "reads as a different quantity.")
    ap.add_argument('--save_samples', action='store_true',
                    help='write each of the K draws as <name>_sample<k>.ply, so the '
                         'diversity figure can be recomposed without re-sampling')
    ap.add_argument('--view', default='xy')
    ap.add_argument('--zoom', type=float, default=1.1)
    ap.add_argument('--no_figures', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    cfg_path = resolve_config_file(args.data, args.config_file)
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    # Match the per-arm checkpoint subfolder written by train_diffusion.py. With
    # no --param this stays on the legacy base dir + eps sampler (old behaviour).
    predict = args.param or dcfg.get('predict', 'eps')
    dcfg['predict'] = predict
    arm = 'base (legacy)'
    if args.param is not None:
        arm = f"predict_{predict}" + (f"_{args.tag}" if args.tag else "")
        config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'config : {cfg_path}')
    print(f'arm    : {arm}   (predict={predict})')
    print('device :', device)

    denoiser = build_denoiser(config, dcfg, device, use_ema=not args.no_ema,
                              use_last=args.use_last)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=predict).to(device)
    template_mesh, _ = get_template_info(config)
    faces = np.array(template_mesh.faces)
    N_t = len(template_mesh.vertices)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    if isinstance(scale, dict):
        sigma, clip_value = scale['sigma'].to(device), scale.get('clip_value', None)
    else:
        sigma, clip_value = scale.to(device), None
    if args.clip is not None:
        clip_value = args.clip
    print(f'clip_value={clip_value}  steps={num_steps}  K={args.num_samples}\n')

    solver = BoundaryFieldSolver(np.asarray(faces, dtype=np.int64),
                                 np.asarray(template_mesh.vertices))
    L = uniform_laplacian(solver._A, N_t)

    partials = sorted(glob(os.path.join(args.partial_dir, '*.ply')))
    rows = []
    metrics = []
    for pf in partials:
        name = os.path.basename(pf).replace('.ply', '')
        gt_path = gt_for(os.path.basename(pf), args.gt_dir)
        if gt_path is None:
            print(f'  {name}: no matching GT, skipping')
            continue

        gt_v = np.array(load_mesh(gt_path, process=False).vertices, dtype=np.float32)
        part_v = np.array(load_mesh(pf, process=False).vertices, dtype=np.float32)
        if len(gt_v) != N_t:
            print(f'  {name}: GT has {len(gt_v)} verts (expected {N_t}), skipping')
            continue

        # A template vertex is KNOWN if it matches a partial vertex (dist ~ 0).
        d, _ = cKDTree(part_v).query(gt_v)
        missing = d > 1e-5                          # True = removed in the partial
        partial_in_template = gt_v.copy()
        partial_in_template[missing] = 0.0

        print(f'  {name}: missing {missing.sum()}/{N_t} ({100*missing.mean():.1f}%)')
        t0 = time.perf_counter()
        samples = complete(denoiser, diffusion, partial_in_template, missing, config,
                           device, sigma, clip_value=clip_value,
                           num_steps=num_steps, num_samples=args.num_samples)
        comp = samples[0]

        out_ply = os.path.join(args.out_dir, f'{name}_completed.ply')
        save_ply_explicit(Trimesh(vertices=comp, faces=faces, process=False), out_ply)

        err = np.linalg.norm(comp - gt_v, axis=1)
        m_err = err[missing].mean() if missing.any() else 0.0

        # ---- Metrics, same definitions as research/diag_sampling_floor.py ----
        gt64 = gt_v.astype(np.float64)
        stack = np.stack([np.asarray(s, dtype=np.float64) for s in samples], axis=0)
        band = boundary_band(solver._A, missing, rings=2)
        per = [mesh_stats(s, gt64, faces, missing, band, L) for s in stack]
        single = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        single_sd = {k: float(np.std([p[k] for p in per])) for k in per[0]}

        mean_mesh = stack.mean(axis=0)
        mean_mesh[~missing] = gt64[~missing]        # known stays exact
        meanK = mesh_stats(mean_mesh, gt64, faces, missing, band, L)
        spread_mm = float(1000.0 * np.linalg.norm(stack.std(axis=0), axis=1)[missing].mean()) \
                    if args.num_samples > 1 else 0.0

        if args.save_samples:
            for k, s in enumerate(stack):
                save_ply_explicit(Trimesh(vertices=s, faces=faces, process=False),
                                  os.path.join(args.out_dir, f'{name}_sample{k+1}.ply'))

        if args.num_samples > 1:
            mean_ply = os.path.join(args.out_dir, f'{name}_completed_meanofK.ply')
            save_ply_explicit(Trimesh(vertices=mean_mesh, faces=faces, process=False), mean_ply)
            # Per-vertex fields, so uncertainty-vs-error can be scored later
            # without re-sampling (see research/analyze_real_partial.py).
            np.savez_compressed(
                os.path.join(args.out_dir, f'{name}_pervertex.npz'),
                missing=missing,
                unc_mm=1000.0 * np.linalg.norm(stack.std(axis=0), axis=1),
                err_first_mm=1000.0 * np.linalg.norm(stack[0] - gt64, axis=1),
                err_meanK_mm=1000.0 * np.linalg.norm(mean_mesh - gt64, axis=1))

        metrics.append({
            'name': name, 'K': args.num_samples,
            'n_missing': int(missing.sum()), 'missing_pct': 100.0 * float(missing.mean()),
            'single_pos_mm': single['pos_mm'], 'single_pos_sd_mm': single_sd['pos_mm'],
            'meanK_pos_mm': meanK['pos_mm'],
            'single_pos_p95_mm': single['pos_p95_mm'], 'single_pos_max_mm': single['pos_max_mm'],
            'single_nrm_deg': single['nrm_deg'], 'meanK_nrm_deg': meanK['nrm_deg'],
            'single_seam_deg': single['seam_deg'], 'meanK_seam_deg': meanK['seam_deg'],
            'single_rough': single['rough_ratio'], 'meanK_rough': meanK['rough_ratio'],
            'spread_mm': spread_mm,
            'firstdraw_pos_mm': float(1000.0 * m_err),
        })
        print(f"      position {single['pos_mm']:.3f} mm (mean-of-K {meanK['pos_mm']:.3f})  "
              f"normals {single['nrm_deg']:.3f} deg  seam {single['seam_deg']:.3f} deg  "
              f"spread {spread_mm:.3f} mm  [{time.perf_counter()-t0:.1f}s]")

        rows.append(dict(name=name, gt_v=gt_v, comp_v=comp, missing=missing,
                         err=err, faces=faces, samples=[np.asarray(s) for s in samples],
                         mean_v=mean_mesh,
                         unc_mm=1000.0 * np.linalg.norm(stack.std(axis=0), axis=1)))

    if not rows:
        print('No meshes processed.')
        return

    # ---- metrics.csv + summary.txt --------------------------------------
    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader(); w.writerows(metrics)

    def fmt_table():
        hdr = (f"{'mesh':<12}{'miss%':>7}{'pos mm':>9}{'meanK':>8}{'p95':>8}"
               f"{'max':>8}{'nrm deg':>9}{'seam':>8}{'spread':>8}")
        lines = [hdr, '-' * len(hdr)]
        for m in metrics:
            lines.append(f"{m['name']:<12}{m['missing_pct']:>7.1f}"
                         f"{m['single_pos_mm']:>9.3f}{m['meanK_pos_mm']:>8.3f}"
                         f"{m['single_pos_p95_mm']:>8.2f}{m['single_pos_max_mm']:>8.2f}"
                         f"{m['single_nrm_deg']:>9.3f}{m['single_seam_deg']:>8.3f}"
                         f"{m['spread_mm']:>8.3f}")
        lines.append('-' * len(hdr))
        lines.append(f"{'MEAN':<12}{np.mean([m['missing_pct'] for m in metrics]):>7.1f}"
                     f"{np.mean([m['single_pos_mm'] for m in metrics]):>9.3f}"
                     f"{np.mean([m['meanK_pos_mm'] for m in metrics]):>8.3f}"
                     f"{np.mean([m['single_pos_p95_mm'] for m in metrics]):>8.2f}"
                     f"{np.mean([m['single_pos_max_mm'] for m in metrics]):>8.2f}"
                     f"{np.mean([m['single_nrm_deg'] for m in metrics]):>9.3f}"
                     f"{np.mean([m['single_seam_deg'] for m in metrics]):>8.3f}"
                     f"{np.mean([m['spread_mm'] for m in metrics]):>8.3f}")
        return '\n'.join(lines)

    summary = (f"arm            : {arm}   (predict={predict})\n"
               f"weights        : {'ema_last' if args.use_last else 'ema (best-eval)'}\n"
               f"partial_dir    : {args.partial_dir}\n"
               f"gt_dir         : {args.gt_dir}\n"
               f"K draws        : {args.num_samples}   DDIM steps: {num_steps}   "
               f"seed: {args.seed}\n"
               f"meshes         : {len(metrics)}\n\n"
               f"Per-vertex L2 over the MISSING region, in mm. 'pos mm' is the expected\n"
               f"error of ONE draw (mean over K); 'meanK' is the error of the average of\n"
               f"the K draws (the empirical conditional mean, not a valid single sample).\n"
               f"'spread' is the mean per-vertex std across draws = model uncertainty.\n\n"
               + fmt_table() + "\n")
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    print('\n' + summary)
    print('wrote', csv_path)

    if args.no_figures:
        print(f'\nDone (figures skipped). Completed meshes in {args.out_dir}')
        return

    # Per-sample figures: GT | Input (hole) | Ours | Error
    vmax = float(np.percentile(np.concatenate([r['err'] for r in rows]), 99))
    clim = [0.0, vmax]
    titles = ["GT", "Input", "Ours", "Error"]

    def render_row(pl, row, r, with_titles):
        known = ~r['missing']
        keep = known[r['faces']].all(axis=1)
        pl.subplot(row, 0)
        add_surface(pl, r['gt_v'], r['faces'], titles[0] if with_titles else "",
                    view=args.view, zoom=args.zoom)
        bold_text(pl, f"{r['name']}  ({100*r['missing'].mean():.0f}% missing)",
                  font_size=18, color='black', position='lower_left')
        pl.subplot(row, 1)
        add_surface(pl, r['gt_v'], r['faces'][keep], titles[1] if with_titles else "",
                    view=args.view, zoom=args.zoom)
        pl.subplot(row, 2)
        add_surface(pl, r['comp_v'], r['faces'], titles[2] if with_titles else "",
                    view=args.view, zoom=args.zoom)
        pl.subplot(row, 3)
        add_surface(pl, r['comp_v'], r['faces'], titles[3] if with_titles else "",
                    scalars=r['err'], clim=clim, show_bar=True,
                    view=args.view, zoom=args.zoom)

    for r in rows:
        pl = pv.Plotter(off_screen=True, shape=(1, 4), window_size=[2000, 560], border=False)
        pl.background_color = 'white'
        render_row(pl, 0, r, with_titles=True)
        out = os.path.join(fig_dir, f"{r['name']}.png")
        pl.screenshot(out); pl.close()
        print('  fig:', out)

    R = len(rows)
    pl = pv.Plotter(off_screen=True, shape=(R, 4), window_size=[2000, 500 * R], border=False)
    pl.background_color = 'white'
    for i, r in enumerate(rows):
        render_row(pl, i, r, with_titles=(i == 0))
    grid = os.path.join(fig_dir, '_grid.png')
    pl.screenshot(grid); pl.close()
    print('  grid:', grid)

    # ---- Diversity + uncertainty figures (only meaningful with num_samples > 1) ----
    if args.num_samples > 1:
        div_dir = os.path.join(fig_dir, 'diversity')
        os.makedirs(div_dir, exist_ok=True)
        K = args.num_samples

        # ONE shared uncertainty scale over every row, in MM -- same convention as
        # every other paper figure (jet, millimetres, one bar). A per-row clim
        # would make the single drawn bar apply to only one row, and the rows
        # would not be comparable to each other.
        unc_all = np.concatenate([r['unc_mm'] for r in rows])
        unc_clim = [0.0, float(np.percentile(unc_all, 99)) or 1e-6]
        print(f'shared uncertainty scale: 0 - {unc_clim[1]:.2f} mm (cmap={args.unc_cmap})')

        def render_div_row(pl, row, r, with_titles, with_bar=True):
            samples = r['samples']
            unc = r['unc_mm']                        # (N,) per-vertex std, mm
            known = ~r['missing']
            keep = known[r['faces']].all(axis=1)
            ncol = 2 + K + 1
            # GT
            pl.subplot(row, 0)
            add_surface(pl, r['gt_v'], r['faces'], "GT" if with_titles else "",
                        view=args.view, zoom=args.zoom)
            bold_text(pl, f"{r['name']} ({100*r['missing'].mean():.0f}% missing)",
                      font_size=18, color='black', position='lower_left')
            # Input
            pl.subplot(row, 1)
            add_surface(pl, r['gt_v'], r['faces'][keep], "Input" if with_titles else "",
                        view=args.view, zoom=args.zoom)
            # K diverse completions
            for k in range(K):
                pl.subplot(row, 2 + k)
                add_surface(pl, samples[k], r['faces'],
                            f"Sample {k+1}" if with_titles else "",
                            view=args.view, zoom=args.zoom)
            # Uncertainty (std across samples) on the mean shape
            mean_v = np.stack(samples, 0).mean(0)
            pl.subplot(row, ncol - 1)
            # The bar is drawn on ONE row only (the middle one for a grid), so it
            # ends up vertically centred on the figure like the colorbar in the
            # ladder / qualitative figures instead of hanging off the top row.
            add_surface(pl, mean_v, r['faces'], "Uncertainty" if with_titles else "",
                        scalars=unc, clim=unc_clim, cmap=args.unc_cmap,
                        show_bar=with_bar, view=args.view, zoom=args.zoom,
                        bar_args=dict(title="std (mm)", vertical=True,
                                      title_font_size=20, label_font_size=18,
                                      position_x=0.80, position_y=0.20,
                                      height=0.60, width=0.05, n_labels=5,
                                      fmt='%.2f', color='black'))

        ncol = 2 + K + 1
        # Per-mesh diversity figures.
        for r in rows:
            pl = pv.Plotter(off_screen=True, shape=(1, ncol),
                            window_size=[420 * ncol, 560], border=False)
            pl.background_color = 'white'
            render_div_row(pl, 0, r, with_titles=True)
            out = os.path.join(div_dir, f"{r['name']}_diversity.png")
            pl.screenshot(out); pl.close()
            print('  diversity fig:', out)

        # Combined diversity grid for the large-hole cases (where ambiguity is real).
        big = [r for r in rows if r['missing'].mean() > 0.25]
        if big:
            pl = pv.Plotter(off_screen=True, shape=(len(big), ncol),
                            window_size=[420 * ncol, 500 * len(big)], border=False)
            pl.background_color = 'white'
            mid = len(big) // 2
            for i, r in enumerate(big):
                render_div_row(pl, i, r, with_titles=(i == 0), with_bar=(i == mid))
            gpath = os.path.join(div_dir, '_diversity_grid_largeholes.png')
            pl.screenshot(gpath); pl.close()
            print('  diversity grid (large holes):', gpath)

    print(f'\nDone. Completed meshes in {args.out_dir}')


if __name__ == '__main__':
    main()
