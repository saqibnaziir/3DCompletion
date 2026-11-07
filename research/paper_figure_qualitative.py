"""
Paper figure: main qualitative completion results (one arm, several meshes).

Layout (one row per mesh):

                    Input | Ours | Ground truth | Error
  patch, 15% missing   .      .         .           .
  plane cut, 12% ...   .      .         .           .

Unlike research/qualitative_figure.py (a quick PyVista contact sheet), panels
here are rendered transparent, given a soft drop shadow, tightly cropped to a
single shared box, and composed in matplotlib -- so the text stays vector in the
PDF, the font matches the CVPR body text, and there is no dead white space
between meshes. The shadow/crop helpers are shared with the ablation figure.

Every error panel shares ONE color scale, in millimetres.

Outputs <out>/qualitative_figure.{pdf,png}.

Run (conda 3dmesh python):
  python research/paper_figure_qualitative.py
  python research/paper_figure_qualitative.py --samples test_012 test_004 test_001
  python research/paper_figure_qualitative.py --clim 6.0 --method_label Completion
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

# Rendering / shadow / crop are identical to the ablation figure -- reuse them
# so the two figures cannot drift apart visually.
from paper_figure_ladder import render, drop_shadow, union_crop, STRAT_PRETTY

TEST_DIR = os.path.join('data', 'completion_test_samples', 'test_set')
ARM = 'completed_diffusion_x0_depthw_nrm_scratch'
SUFFIX = '_completed_diffusion.ply'

# CVPR body text is Times; match it so figure labels do not read as a foreign
# element on the page. fonttype 42 embeds TrueType (required by some venues).
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'Times',
                   'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def fit_zoom(rgba, margin=0.97):
    """Largest camera zoom that still keeps this panel's mesh inside the canvas.

    render() fits the mesh with reset_camera() and then zooms in, which crops
    whatever falls outside the viewport -- for a standing body that is the head
    and the feet. Zoom scales about the canvas centre, so the headroom is the
    smaller of (half-canvas / farthest content offset) per axis.
    """
    h, w = rgba.shape[:2]
    rs, cs = np.where(rgba[..., 3] > 0)
    if len(rs) == 0:
        return np.inf
    cy, cx = h / 2.0, w / 2.0
    dy = max(cy - rs.min(), rs.max() + 1 - cy)
    dx = max(cx - cs.min(), cs.max() + 1 - cx)
    return margin * min(cy / dy, cx / dx)


def load_sample(name, gt_dir, mask_dir, comp_dir):
    gt = load_mesh(os.path.join(gt_dir, f'{name}_gt_complete.ply'), process=False)
    gt_v = np.asarray(gt.vertices, dtype=np.float64)
    faces = np.asarray(gt.faces)
    mask = np.load(os.path.join(mask_dir, f'{name}_mask.npy'))     # True = missing

    comp_path = os.path.join(comp_dir, f'{name}{SUFFIX}')
    if not os.path.exists(comp_path):
        raise SystemExit(f'missing completion: {comp_path}')
    comp_v = np.asarray(load_mesh(comp_path, process=False).vertices, dtype=np.float64)

    partials = glob(os.path.join(mask_dir, f'{name}_partial_*.ply'))
    strat = 'unknown'
    if partials:
        raw = os.path.basename(partials[0])
        raw = raw.replace(f'{name}_partial_', '').replace('.ply', '').split('_r')[0]
        strat = STRAT_PRETTY.get(raw, raw.replace('_', ' '))

    keep = (~mask)[faces].all(axis=1)              # faces fully inside the known region
    err = np.linalg.norm(comp_v - gt_v, axis=1) * 1000.0        # m -> mm

    return dict(name=name, gt_v=gt_v, faces=faces, mask=mask, comp_v=comp_v,
                partial_faces=faces[keep], err=err, strat=strat)


def main():
    ap = argparse.ArgumentParser()
    # Six meshes covering all three masking strategies; 2 blocks x 3 rows fills
    # exactly. Avoid masks that are a whole eyeball (a disconnected component of
    # the COMA template): invisible from --view, and the error map reads all-blue.
    ap.add_argument('--samples', nargs='*',
                    default=['test_012', 'test_006', 'test_013',
                             'test_004', 'test_010', 'test_001'],
                    help='mesh ids, one row each')
    ap.add_argument('--blocks', type=int, default=2,
                    help='side-by-side column groups. 1 = single tall stack '
                         '(only usable up to ~3 samples on a letter page)')
    ap.add_argument('--test_dir', default=TEST_DIR)
    ap.add_argument('--arm', default=ARM, help='completion folder under --test_dir')
    ap.add_argument('--gt_dir', default=None)
    ap.add_argument('--mask_dir', default=None)
    ap.add_argument('--comp_dir', default=None)
    ap.add_argument('--out_dir', default=None, help='default: <test_dir>/paper_figures')
    ap.add_argument('--clim', type=float, default=None,
                    help='max of the shared error scale, in MM (default: p99)')
    ap.add_argument('--cmap', default='jet')
    ap.add_argument('--cbar_frac', type=float, default=0.62,
                    help='colorbar height as a fraction of the panel block')
    ap.add_argument('--view', default='xy')
    ap.add_argument('--zoom', type=float, default=1.5,
                    help='larger = mesh fills more of its panel (upper bound '
                         'only, unless --no_autofit)')
    ap.add_argument('--no_autofit', dest='autofit', action='store_false',
                    help='trust --zoom even if it crops meshes at the canvas edge')
    ap.add_argument('--method_label', default='Ours',
                    help='header over the completion column')
    ap.add_argument('--no_shadow', action='store_true')
    ap.add_argument('--shadow_blur', type=float, default=9.0)
    ap.add_argument('--shadow_strength', type=float, default=0.42)
    ap.add_argument('--shadow_offset', type=int, nargs=2, default=(14, 10),
                    metavar=('DOWN', 'RIGHT'))
    ap.add_argument('--width_in', type=float, default=7.16,
                    help='7.16 = CVPR two-column span')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--hspace', type=float, default=0.012,
                    help='vertical gap between rows, in panel heights')
    ap.add_argument('--wspace', type=float, default=0.012,
                    help='horizontal gap between columns, in panel widths')
    ap.add_argument('--title_pad', type=float, default=0.20,
                    help='inches reserved above the grid for the column titles')
    ap.add_argument('--title_gap', type=float, default=2.5,
                    help='points between a column title and its panel')
    ap.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                    help='output extensions, e.g. --formats jpg')
    args = ap.parse_args()

    gt_dir = args.gt_dir or os.path.join(args.test_dir, 'gt')
    mask_dir = args.mask_dir or os.path.join(args.test_dir, 'partial')
    comp_dir = args.comp_dir or os.path.join(args.test_dir, args.arm)
    out_dir = args.out_dir or os.path.join(args.test_dir, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    samples = [load_sample(n, gt_dir, mask_dir, comp_dir) for n in args.samples]

    all_err = np.concatenate([s['err'] for s in samples])
    vmax = args.clim if args.clim is not None else float(np.percentile(all_err, 99))
    clim = (0.0, vmax)
    print(f'shared error scale: 0 - {vmax:.2f} mm  (cmap={args.cmap})')

    # ---- render every panel ------------------------------------------------
    def render_row(s, zoom):
        return [
            render(s['gt_v'], s['partial_faces'], zoom=zoom, view=args.view),
            render(s['comp_v'], s['faces'], zoom=zoom, view=args.view),
            render(s['gt_v'], s['faces'], zoom=zoom, view=args.view),
            render(s['comp_v'], s['faces'], scalars=s['err'], clim=clim,
                   cmap=args.cmap, zoom=zoom, view=args.view),
        ]

    zoom = args.zoom
    if args.autofit:
        # Probe pass at zoom 1 (reset_camera alone never clips), then take the
        # tightest per-panel headroom as ONE shared zoom -- a per-panel zoom
        # would break the constant head scale the union crop is there to keep.
        probe = [fit_zoom(p) for s in samples for p in render_row(s, 1.0)]
        zoom = min(args.zoom, float(min(probe)))
        if zoom < args.zoom:
            print(f'  autofit: zoom {args.zoom:.2f} would clip -> {zoom:.2f}')

    for s in samples:
        print(f"  rendering {s['name']} ({s['strat']}, "
              f"{100 * s['mask'].mean():.0f}% missing) ...")
        panels = render_row(s, zoom)
        if args.no_shadow:
            panels = [p[..., :3] if p.shape[2] == 4 else p for p in panels]
        else:
            panels = [drop_shadow(p, offset=tuple(args.shadow_offset),
                                  blur=args.shadow_blur,
                                  strength=args.shadow_strength)
                      for p in panels]
        s['panels'] = panels

    # ONE crop box for the whole figure: identical pixel dimensions everywhere,
    # so matplotlib never letterboxes and head scale is constant across rows.
    ncol = 4
    flat = union_crop([p for s in samples for p in s['panels']])
    for i, s in enumerate(samples):
        s['panels'] = flat[i * ncol:(i + 1) * ncol]

    # ---- compose -----------------------------------------------------------
    # Panels are laid out in `blocks` side-by-side column groups. A single
    # 4-column stack grows ~2.4 in per mesh, so beyond three meshes it no longer
    # fits the text block of a letter page; two blocks halve the height.
    ph, pw = samples[0]['panels'][0].shape[:2]
    n = len(samples)
    nb = max(1, args.blocks)
    nrow = int(np.ceil(n / nb))

    L, R = 0.006, 0.928            # right margin leaves room for the colorbar
    GAP = 0.030                    # between blocks
    GUT = 0.021                    # row-label gutter inside each block
    block_w = (R - L - GAP * (nb - 1)) / nb

    # gridspec spends wspace/hspace *inside* the block, so account for the gaps
    # here -- otherwise wider gaps silently shrink the meshes instead of making
    # the figure taller.
    panel_w_in = ((block_w - GUT) / (ncol + (ncol - 1) * args.wspace)
                  * args.width_in)
    panel_h_in = panel_w_in * ph / pw
    top_pad, bot_pad = args.title_pad, 0.04       # inches: column titles / trim
    fig_h = panel_h_in * (nrow + (nrow - 1) * args.hspace) + top_pad + bot_pad

    fig = plt.figure(figsize=(args.width_in, fig_h))
    top_f, bot_f = 1.0 - top_pad / fig_h, bot_pad / fig_h

    col_titles = ['Input', args.method_label, 'Ground truth', 'Error']

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    grids = []
    for b in range(nb):
        left = L + b * (block_w + GAP)
        grids.append(gridspec.GridSpec(nrow, ncol, figure=fig,
                                       left=left + GUT, right=left + block_w,
                                       top=top_f, bottom=bot_f,
                                       wspace=args.wspace, hspace=args.hspace))

    for si, s in enumerate(samples):
        b, r = divmod(si, nrow)                   # fill each block top to bottom
        gs = grids[b]
        for c in range(ncol):
            ax = bare(fig.add_subplot(gs[r, c]))
            ax.imshow(s['panels'][c])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7.6, pad=args.title_gap)

        # Row label in the block's gutter: masking strategy + missing ratio only.
        pos = gs[r, 0].get_position(fig)
        fig.text(L + b * (block_w + GAP) + GUT * 0.42,
                 pos.y0 + pos.height / 2,
                 f"{s['strat']}, {100 * s['mask'].mean():.0f}% missing",
                 rotation=90, ha='center', va='center', fontsize=6.6, color='0.25')

    # Shared vertical colorbar, centred on the panel block. Spanning the full
    # height makes it read as an extra column and spreads the ticks too far.
    y_top = grids[0][0, 0].get_position(fig).y1
    y_bot = grids[0][nrow - 1, 0].get_position(fig).y0
    h = (y_top - y_bot) * args.cbar_frac
    cax = fig.add_axes([0.940, (y_top + y_bot) / 2 - h / 2, 0.014, h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=args.cmap),
                      cax=cax, orientation='vertical')
    cb.set_label('per-vertex error (mm)', fontsize=7.0, labelpad=3, rotation=90)
    cb.ax.tick_params(labelsize=6.4, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)

    for ext in args.formats:
        out = os.path.join(out_dir, f'qualitative_figure.{ext}')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
