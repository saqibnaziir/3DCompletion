"""
Paper figure: one mesh, every method side by side, error map under each.

Layout (2 rows x 4 columns):

    Ground truth |   FaceCom    | 3DGeoMeshNet |     Ours
    Input        |  error map   |  error map   |  error map

The top-left panel is the ground truth and the bottom-left is the partial input
that every method was given, so the reader can see the hole being filled next to
the reference. The remaining three columns are one method each: completion on
top, per-vertex error below, all three sharing ONE color scale in millimetres.

Rendering, drop shadow and cropping are imported from paper_figure_ladder so
this figure cannot drift visually from qualitative_figure / ladder_figure.

SAMPLE CHOICE. A single mesh can be made to favour any method -- on patch masks
the per-sample spread between methods is larger than the gap between their
averages. So the default is not hand-picked: --pick representative takes the
sample whose per-method errors sit closest (in relative terms) to each method's
average over that masking strategy, which is the one that misleads least. Use
--sample to override, and say in the caption which rule you used.

Outputs <out>/baseline_qualitative.{pdf,png}.

Run (conda 3dmesh python):
  python scripts/paper_figure_baselines.py
  python scripts/paper_figure_baselines.py --strategy random_scatter
  python scripts/paper_figure_baselines.py --sample test_013 --clim 8
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from glob import glob

import numpy as np
from trimesh import load_mesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec

from paper_figure_ladder import render, drop_shadow, union_crop, STRAT_PRETTY

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join('data', 'completion_test_samples', 'test_set')

# label -> (directory, filename suffix). Order is the column order.
METHODS = [
    ('FaceCom', os.path.join('Compattators', 'FaceCom_repo',
                             'completed_facecom_mm'), '_completed_facecom.ply'),
    ('3DGeoMeshNet', os.path.join('Compattators', '3DGeoMeshNet',
                                  'completed_fit_pipeline'),
     '_completed_3dgeomeshnet.ply'),
    ('Ours', os.path.join(TEST_DIR, 'completed_diffusion_x0_depthw_nrm_scratch'),
     '_completed_diffusion.ply'),
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'Times',
                   'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def fit_zoom(rgba, margin=0.97):
    """Largest zoom that still keeps this panel's mesh inside the canvas."""
    h, w = rgba.shape[:2]
    rs, cs = np.where(rgba[..., 3] > 0)
    if len(rs) == 0:
        return np.inf
    cy, cx = h / 2.0, w / 2.0
    dy = max(cy - rs.min(), rs.max() + 1 - cy)
    dx = max(cx - cs.min(), cs.max() + 1 - cx)
    return margin * min(cy / dy, cx / dx)


def strategy_of(test_dir, base):
    hits = glob(os.path.join(test_dir, 'partial', f'{base}_partial_*.ply'))
    if not hits:
        return 'unknown'
    name = os.path.basename(hits[0])
    for t in STRAT_PRETTY:
        if t in name:
            return t
    return 'unknown'


def gather(test_dir, strategy, max_missing):
    """Per-sample missing ratio and per-method hole error for one strategy."""
    rows = []
    for gt_file in sorted(glob(os.path.join(test_dir, 'gt', '*_gt_complete.ply'))):
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        if strategy and strategy_of(test_dir, base) != strategy:
            continue
        mask = np.load(os.path.join(test_dir, 'partial',
                                    f'{base}_mask.npy')).astype(bool)
        if max_missing and 100.0 * mask.mean() > max_missing:
            continue
        gt = np.asarray(load_mesh(gt_file, process=False).vertices,
                        dtype=np.float64)
        errs = {}
        for label, d, suffix in METHODS:
            p = os.path.join(PROJECT_ROOT, d, base + suffix)
            if not os.path.exists(p):
                errs = None
                break
            v = np.asarray(load_mesh(p, process=False).vertices, dtype=np.float64)
            errs[label] = float(np.linalg.norm(v - gt, axis=1)[mask].mean() * 1000)
        if errs:
            rows.append(dict(base=base, missing=100.0 * mask.mean(), errs=errs))
    return rows


def pick_representative(rows):
    """Sample whose per-method errors are relatively closest to the group means.

    Relative, not absolute: an absolute criterion would just pick whichever mesh
    is easiest overall. Scoring each method's deviation as a ratio to its own
    mean keeps the choice about the *pattern* between methods.
    """
    labels = [m[0] for m in METHODS]
    mean = {k: np.mean([r['errs'][k] for r in rows]) for k in labels}
    def score(r):
        return sum(abs(np.log(r['errs'][k] / mean[k])) for k in labels)
    return min(rows, key=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', default=TEST_DIR)
    ap.add_argument('--sample', default=None,
                    help='mesh id, e.g. test_013. Overrides --pick')
    ap.add_argument('--strategy', default='bfs_patch',
                    choices=['bfs_patch', 'plane_cut', 'random_scatter', ''],
                    help="masking strategy to choose from ('' = any)")
    ap.add_argument('--max_missing', type=float, default=12.0,
                    help='only consider samples up to this %% missing')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--clim', type=float, default=None,
                    help='max of the shared error scale in MM (default: p99 '
                         'over the three error maps)')
    ap.add_argument('--cmap', default='jet')
    ap.add_argument('--view', default='xy')
    ap.add_argument('--zoom', type=float, default=1.5)
    ap.add_argument('--no_autofit', dest='autofit', action='store_false')
    ap.add_argument('--no_shadow', action='store_true')
    ap.add_argument('--shadow_blur', type=float, default=9.0)
    ap.add_argument('--shadow_strength', type=float, default=0.42)
    ap.add_argument('--shadow_offset', type=int, nargs=2, default=(14, 10))
    ap.add_argument('--width_in', type=float, default=7.16,
                    help='7.16 = CVPR two-column span')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--hspace', type=float, default=0.012)
    ap.add_argument('--wspace', type=float, default=0.012)
    ap.add_argument('--title_pad', type=float, default=0.20)
    ap.add_argument('--title_gap', type=float, default=2.5)
    ap.add_argument('--annotate', action='store_true',
                    help='print each error map\'s mean error under the panel')
    ap.add_argument('--out_name', default='baseline_qualitative',
                    help='output basename, so several samples can coexist')
    ap.add_argument('--formats', nargs='+', default=['pdf', 'png'])
    args = ap.parse_args()

    test_dir = args.test_dir
    out_dir = args.out_dir or os.path.join(test_dir, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    rows = gather(test_dir, args.strategy, args.max_missing)
    if not rows:
        raise SystemExit('no sample matched --strategy/--max_missing with all '
                         'three completions present')
    if args.sample:
        hit = [r for r in rows if r['base'] == args.sample]
        if not hit:                       # allow an explicit sample outside the filter
            hit = gather(test_dir, '', None)
            hit = [r for r in hit if r['base'] == args.sample]
        if not hit:
            raise SystemExit(f'{args.sample}: not found, or a completion is missing')
        chosen = hit[0]
    else:
        chosen = pick_representative(rows)

    base = chosen['base']
    print(f"candidates ({args.strategy or 'any'}, <= {args.max_missing}% missing): "
          f"{', '.join(r['base'] for r in rows)}")
    print(f"chosen: {base}  ({chosen['missing']:.1f}% missing)")
    for k, v in chosen['errs'].items():
        print(f"    {k:<14} {v:7.3f} mm")

    # ---- load geometry ------------------------------------------------------
    gt_mesh = load_mesh(os.path.join(test_dir, 'gt', f'{base}_gt_complete.ply'),
                        process=False)
    gt_v = np.asarray(gt_mesh.vertices, dtype=np.float64)
    faces = np.asarray(gt_mesh.faces)
    mask = np.load(os.path.join(test_dir, 'partial', f'{base}_mask.npy')).astype(bool)
    keep = (~mask)[faces].all(axis=1)          # faces wholly inside the known region

    comps, errs = [], []
    for label, d, suffix in METHODS:
        v = np.asarray(load_mesh(os.path.join(PROJECT_ROOT, d, base + suffix),
                                 process=False).vertices, dtype=np.float64)
        comps.append(v)
        errs.append(np.linalg.norm(v - gt_v, axis=1) * 1000.0)

    vmax = args.clim if args.clim is not None else float(
        np.percentile(np.concatenate(errs), 99))
    clim = (0.0, vmax)
    print(f'shared error scale: 0 - {vmax:.2f} mm  (cmap={args.cmap})')

    # ---- render -------------------------------------------------------------
    def panels_at(zoom):
        top = [render(gt_v, faces, zoom=zoom, view=args.view)]
        top += [render(v, faces, zoom=zoom, view=args.view) for v in comps]
        bot = [render(gt_v, faces[keep], zoom=zoom, view=args.view)]
        bot += [render(v, faces, scalars=e, clim=clim, cmap=args.cmap,
                       zoom=zoom, view=args.view) for v, e in zip(comps, errs)]
        return top + bot

    zoom = args.zoom
    if args.autofit:
        zoom = min(args.zoom, float(min(fit_zoom(p) for p in panels_at(1.0))))
        if zoom < args.zoom:
            print(f'  autofit: zoom {args.zoom:.2f} would clip -> {zoom:.2f}')

    print(f'  rendering {base} ...')
    panels = panels_at(zoom)
    if args.no_shadow:
        panels = [p[..., :3] if p.shape[2] == 4 else p for p in panels]
    else:
        panels = [drop_shadow(p, offset=tuple(args.shadow_offset),
                              blur=args.shadow_blur,
                              strength=args.shadow_strength) for p in panels]
    panels = union_crop(panels)                # one crop box -> constant head scale

    # ---- compose ------------------------------------------------------------
    ncol, nrow = 4, 2
    ph, pw = panels[0].shape[:2]
    L, R = 0.006, 0.928                        # right margin holds the colorbar
    panel_w_in = ((R - L) / (ncol + (ncol - 1) * args.wspace)) * args.width_in
    panel_h_in = panel_w_in * ph / pw
    top_pad, bot_pad = args.title_pad, 0.04
    fig_h = panel_h_in * (nrow + (nrow - 1) * args.hspace) + top_pad + bot_pad

    fig = plt.figure(figsize=(args.width_in, fig_h))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, left=L, right=R,
                           top=1.0 - top_pad / fig_h, bottom=bot_pad / fig_h,
                           wspace=args.wspace, hspace=args.hspace)

    titles = ['Ground truth'] + [m[0] for m in METHODS]
    for i, p in enumerate(panels):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[r, c])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.imshow(p)
        if r == 0:
            ax.set_title(titles[c], fontsize=7.6, pad=args.title_gap)
        elif c == 0:
            ax.set_title(f"Input, {STRAT_PRETTY.get(strategy_of(test_dir, base), '')}"
                         f", {100 * mask.mean():.0f}% missing",
                         fontsize=6.6, color='0.25', pad=args.title_gap)
        elif args.annotate:
            ax.set_xlabel(f'{errs[c - 1][mask].mean():.2f} mm',
                          fontsize=6.6, color='0.25', labelpad=1.5)

    y_top = gs[0, 0].get_position(fig).y1
    y_bot = gs[nrow - 1, 0].get_position(fig).y0
    h = (y_top - y_bot) * 0.62
    cax = fig.add_axes([0.940, (y_top + y_bot) / 2 - h / 2, 0.014, h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim),
                                        cmap=args.cmap),
                      cax=cax, orientation='vertical')
    cb.set_label('per-vertex error (mm)', fontsize=7.0, labelpad=3, rotation=90)
    cb.ax.tick_params(labelsize=6.4, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)

    for ext in args.formats:
        out = os.path.join(out_dir, f'{args.out_name}.{ext}')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
