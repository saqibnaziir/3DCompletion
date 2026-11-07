"""
Paper figure: robustness to half-missing input (no_half / no_half2).

Layout -- one row per crop, GT shown ONCE because both crops were cut from the
same scan (verified byte-identical):

                Input | Ground truth |  Completion | Error
  no_half    partial  |              |   shaded    |  err
  no_half2   partial  |   (spans)    |   shaded    |  err

Every error panel shares ONE color scale + one vertical colorbar, so the two
rows are directly comparable. Panels are rendered off-screen with PyVista and
composed in matplotlib -- same helpers, sizes and fonts as
research/paper_figure_ladder.py, so this figure sits next to that one in the
paper without a visible style break.

Outputs <out_dir>/half_missing_figure.{pdf,png}.

Run (conda 3dmesh python):
  python research/figure_half_missing.py
  python research/figure_half_missing.py --clim 6.0 --suffix _completed_meanofK.ply
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
from trimesh import load_mesh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec

from research.complete_real_partial import gt_for
from research.paper_figure_ladder import render, drop_shadow, union_crop

TEST_OLD = os.path.join('data', 'completion_test_samples', 'test_set', 'Test_old')
COMP_DIR = os.path.join(TEST_OLD, 'completed_x0_depthw_nrm_scratch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', default=['no_half', 'no_half2'])
    ap.add_argument('--comp_dir', default=COMP_DIR)
    ap.add_argument('--gt_dir', default=os.path.join(TEST_OLD, 'gt'))
    ap.add_argument('--partial_dir', default=os.path.join(TEST_OLD, 'partial'))
    ap.add_argument('--out_dir', default=None, help='default: <comp_dir>')
    ap.add_argument('--suffix', default='_completed.ply',
                    help='_completed.ply = one draw (what a user gets); '
                         '_completed_meanofK.ply = the mean of the K draws')
    ap.add_argument('--clim', type=float, default=None,
                    help='max of the shared error scale, in MM (default: p99)')
    ap.add_argument('--cmap', default='jet')
    ap.add_argument('--cbar_frac', type=float, default=0.55,
                    help='colorbar height as a fraction of the panel block')
    ap.add_argument('--view', default='xy')
    ap.add_argument('--zoom', type=float, default=1.45)
    ap.add_argument('--no_shadow', action='store_true')
    ap.add_argument('--shadow_blur', type=float, default=9.0)
    ap.add_argument('--shadow_strength', type=float, default=0.42)
    ap.add_argument('--shadow_offset', type=int, nargs=2, default=(14, 10),
                    metavar=('DOWN', 'RIGHT'))
    ap.add_argument('--width_in', type=float, default=7.16,
                    help='figure width in inches (7.16 = CVPR two-column span)')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    out_dir = args.out_dir or args.comp_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- load -------------------------------------------------------------
    rows, gt_v, faces = [], None, None
    for name in args.names:
        gt_path = gt_for(f'{name}.ply', args.gt_dir)
        part_path = os.path.join(args.partial_dir, f'{name}.ply')
        comp_path = os.path.join(args.comp_dir, f'{name}{args.suffix}')
        for p in (gt_path, part_path, comp_path):
            if p is None or not os.path.exists(p):
                raise SystemExit(f'missing input for {name}: {p}')

        gt_mesh = load_mesh(gt_path, process=False)
        g = np.asarray(gt_mesh.vertices, dtype=np.float64)
        if gt_v is None:
            gt_v, faces = g, np.asarray(gt_mesh.faces)
        elif not np.allclose(g, gt_v):
            raise SystemExit(
                f'{name} has a different GT -- this layout shows GT once and '
                'assumes every crop came from the same scan.')

        part = np.asarray(load_mesh(part_path, process=False).vertices, dtype=np.float64)
        comp = np.asarray(load_mesh(comp_path, process=False).vertices, dtype=np.float64)
        missing = cKDTree(part).query(gt_v)[0] > 1e-5
        err = np.linalg.norm(comp - gt_v, axis=1) * 1000.0          # m -> mm

        rows.append(dict(name=name, comp=comp, missing=missing, err=err,
                         partial_faces=faces[(~missing)[faces].all(axis=1)],
                         mm=float(err[missing].mean())))
        print(f"  {name}: {100 * missing.mean():.1f}% missing, "
              f"hole error {rows[-1]['mm']:.2f} mm")

    # ---- ONE shared color scale over both error panels --------------------
    all_err = np.concatenate([r['err'] for r in rows])
    vmax = args.clim if args.clim is not None else float(np.percentile(all_err, 99))
    clim = (0.0, vmax)
    print(f'shared error scale: 0 - {vmax:.2f} mm  (cmap={args.cmap})')

    # ---- render -----------------------------------------------------------
    def R(verts, f, scalars=None):
        return render(verts, f, scalars=scalars,
                      clim=clim if scalars is not None else None,
                      cmap=args.cmap, zoom=args.zoom, view=args.view)

    print('  rendering ...')
    panels = [R(gt_v, faces)]                                   # [0] = GT, shared
    for r in rows:
        panels += [R(gt_v, r['partial_faces']),                 # input
                   R(r['comp'], faces),                         # completion
                   R(r['comp'], faces, scalars=r['err'])]       # error

    if args.no_shadow:
        panels = [p[..., :3] if p.shape[2] == 4 else p for p in panels]
    else:
        panels = [drop_shadow(p, offset=tuple(args.shadow_offset),
                              blur=args.shadow_blur, strength=args.shadow_strength)
                  for p in panels]
    # One crop box for every panel -> identical pixel size, identical head scale.
    panels = union_crop(panels)
    gt_panel, panels = panels[0], panels[1:]
    for i, r in enumerate(rows):
        r['panels'] = panels[3 * i:3 * i + 3]

    # ---- compose ----------------------------------------------------------
    ph, pw = gt_panel.shape[:2]
    n, ncol = len(rows), 4
    panel_w = args.width_in / (ncol + 0.30)          # 0.30 -> left label gutter
    panel_h = panel_w * ph / pw

    LBL = 0.20                                       # spacer row for the mm labels
    height_ratios = []
    row_of = []
    for _ in range(n):
        row_of.append(len(height_ratios))
        height_ratios += [1.0, LBL]
    nrow = len(height_ratios)
    fig_h = (n + n * LBL) * panel_h + 0.24           # + column titles

    fig = plt.figure(figsize=(args.width_in, fig_h))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, height_ratios=height_ratios,
                           left=0.034, right=0.935, top=0.945, bottom=0.012,
                           wspace=0.02, hspace=0.0)

    col_titles = ['Ground truth', 'Input (partial)', 'Completion', 'Error']

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    # GT leads and spans every row: it is the same mesh for both crops, so
    # repeating it would imply two different targets. Its axes shrinks to the
    # image aspect and is centred, which would drag set_title() down to the
    # middle of the figure -- so its title is drawn as figure text on the
    # column-title line instead.
    ax_gt = bare(fig.add_subplot(gs[:, 0]))
    ax_gt.imshow(gt_panel)
    ax_gt.set_anchor('C')

    for si, r in enumerate(rows):
        rr = row_of[si]
        inp, comp, err = r['panels']
        for c, img in ((1, inp), (2, comp), (3, err)):
            ax = bare(fig.add_subplot(gs[rr, c]))
            # aspect='auto' fills the cell exactly. Safe here because every
            # panel shares one crop box and the cell was sized from that same
            # aspect ratio, so nothing is stretched -- and it keeps the axes
            # box where the gridspec put it, so the titles stay on one line.
            ax.imshow(img, aspect='auto')
            if si == 0:
                ax.set_title(col_titles[c], fontsize=8, pad=3)

        # "% missing" under the input, "x.xx mm" under the error panel. Both are
        # figure-level text so they survive sitting in the spacer row.
        for c, txt, col in ((1, f"{100 * r['missing'].mean():.0f}% missing", '0.2'),
                            (3, f"{r['mm']:.2f} mm", '0.15')):
            p = gs[rr, c].get_position(fig)
            fig.text(p.x0 + p.width / 2, p.y0 - 0.004, txt,
                     ha='center', va='top', fontsize=7.4, color=col)

        # row label in the left gutter
        pos = gs[rr, 0].get_position(fig)
        fig.text(0.011, pos.y0 + pos.height / 2, r['name'].replace('_', r'\_')
                 if matplotlib.rcParams['text.usetex'] else r['name'],
                 rotation=90, ha='center', va='center', fontsize=6.8, color='0.45')

    p_title = gs[row_of[0], 0].get_position(fig)
    fig.text(p_title.x0 + p_title.width / 2, p_title.y1 + 3.0 / 72.0 / fig_h,
             col_titles[0], ha='center', va='bottom', fontsize=8)

    # Shared vertical colorbar, centred on the panel block. Deliberately shorter
    # than the block: a full-height bar reads as a fifth column.
    y_top = gs[row_of[0], 0].get_position(fig).y1
    y_bot = gs[row_of[-1], 0].get_position(fig).y0
    span = y_top - y_bot
    cb_h = args.cbar_frac * span
    cax = fig.add_axes([0.945, y_bot + (span - cb_h) / 2, 0.013, cb_h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=args.cmap),
                      cax=cax, orientation='vertical')
    cb.set_label('per-vertex error (mm)', fontsize=7.5, labelpad=4, rotation=90)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)

    for ext in ('pdf', 'png'):
        out = os.path.join(out_dir, f'half_missing_figure.{ext}')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
