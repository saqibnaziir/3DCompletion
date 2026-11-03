"""
Paper figure: the ablation ladder, side by side on the same meshes.

Layout (one column group per method, two rows per mesh):

               Input | GT | eps | x0 | x0+depth | x0+depth+normal (Ours)
  mesh A  shaded       .     .     .      .          .
          error        -     -     .      .          .
  mesh B  shaded     ...

Every error panel shares ONE color scale, so the panels are directly
comparable -- unlike scripts/qualitative_figure.py, which auto-scales each arm
to its own p99 and therefore must NOT be used for cross-method comparison.

The 3D panels are rendered off-screen with PyVista, then composed in
matplotlib so the text stays vector in the PDF and the colorbar is shared.

Outputs <out>/ladder_figure.{pdf,png}.

Run (conda 3dmesh python):
  python scripts/paper_figure_ladder.py
  python scripts/paper_figure_ladder.py --samples test_013 test_004 --clim 6.0
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from glob import glob

import numpy as np
import pyvista as pv
from trimesh import load_mesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec

TEST_DIR = os.path.join('data', 'completion_test_samples', 'test_set')

# (column label, folder). Order = the ablation ladder, weakest first.
# NOTE: every arm here is our own model -- do NOT tag any column "Ours".
# These are configurations of one method, not us-versus-baselines.
ARMS = [
    (r'$\epsilon$-pred',   'completed_diffusion_ ε'),
    (r'$x_0$-pred',        'completed_diffusion_x0'),
    (r'+ depth ramp',      'completed_diffusion_x0_depthw'),
    (r'+ normal loss',     'completed_diffusion_x0_depthw_nrm_scratch'),
]
SUFFIX = '_completed_diffusion.ply'

STRAT_PRETTY = {'bfs_patch': 'patch', 'plane_cut': 'plane cut',
                'random_scatter': 'scatter'}


def pv_faces(faces):
    return np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64)


def render(verts, faces, scalars=None, clim=None, cmap='jet',
           size=(760, 900), zoom=1.45, view='xy'):
    """Render one panel off-screen -> (H,W,4) uint8 RGBA on a transparent bg.

    Transparent (not white) so the alpha channel gives us the mesh silhouette,
    which drop_shadow() then blurs into the shadow.
    """
    pl = pv.Plotter(off_screen=True, window_size=list(size), border=False)
    mesh = pv.PolyData(verts, pv_faces(faces))
    if scalars is None:
        pl.add_mesh(mesh, color='#d8d8d8', smooth_shading=True,
                    specular=0.35, specular_power=18, show_edges=False)
    else:
        mesh.point_data['err'] = scalars
        pl.add_mesh(mesh, scalars='err', cmap=cmap, clim=clim,
                    smooth_shading=True, show_edges=False,
                    show_scalar_bar=False)
    pl.camera_position = view
    pl.reset_camera()
    pl.camera.zoom(zoom)
    pl.enable_anti_aliasing('ssaa')
    img = pl.screenshot(return_img=True, transparent_background=True)
    pl.close()
    img = np.asarray(img)
    if img.shape[2] == 3:                       # no alpha -> synthesise one
        a = (img < 250).any(axis=2).astype(np.uint8) * 255
        img = np.dstack([img, a])
    return img


def drop_shadow(rgba, offset=(14, 10), blur=9.0, strength=0.42, pad=42):
    """Composite RGBA over white, with the mesh's own blurred silhouette behind it.

    offset is (down, right) in pixels. `pad` keeps the shadow from being clipped
    at the panel edge; the later crop trims whatever space is unused.
    """
    from scipy.ndimage import gaussian_filter

    rgba = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    alpha = rgba[..., 3].astype(np.float64) / 255.0

    dy, dx = offset
    sil = np.roll(np.roll(alpha, dy, axis=0), dx, axis=1)
    shadow = gaussian_filter(sil, blur) * strength

    out = np.ones_like(rgb)                                   # white paper
    out = out * (1.0 - shadow[..., None])                     # darken -> shadow
    out = rgb * alpha[..., None] + out * (1.0 - alpha[..., None])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def content_bbox(img, tol=248):
    """Tight bbox of non-white pixels."""
    nz = np.where((img < tol).any(axis=2))
    if len(nz[0]) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return nz[0].min(), nz[0].max() + 1, nz[1].min(), nz[1].max() + 1


def union_crop(imgs, pad=8):
    """One crop box for a whole mesh group, so every panel keeps the same scale."""
    boxes = [content_bbox(i) for i in imgs]
    r0 = max(0, min(b[0] for b in boxes) - pad)
    r1 = min(imgs[0].shape[0], max(b[1] for b in boxes) + pad)
    c0 = max(0, min(b[2] for b in boxes) - pad)
    c1 = min(imgs[0].shape[1], max(b[3] for b in boxes) + pad)
    return [i[r0:r1, c0:c1] for i in imgs]


def load_sample(name, gt_dir, mask_dir):
    gt = load_mesh(os.path.join(gt_dir, f'{name}_gt_complete.ply'), process=False)
    gt_v = np.asarray(gt.vertices, dtype=np.float64)
    faces = np.asarray(gt.faces)
    mask = np.load(os.path.join(mask_dir, f'{name}_mask.npy'))

    partials = glob(os.path.join(mask_dir, f'{name}_partial_*.ply'))
    strat = 'unknown'
    if partials:
        raw = os.path.basename(partials[0])
        raw = raw.replace(f'{name}_partial_', '').replace('.ply', '').split('_r')[0]
        strat = STRAT_PRETTY.get(raw, raw)

    keep = (~mask)[faces].all(axis=1)          # faces fully in the known region
    return dict(name=name, gt_v=gt_v, faces=faces, mask=mask,
                partial_faces=faces[keep], strat=strat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='*', default=['test_013', 'test_010'],
                    help='mesh ids, one row-group each. Pick meshes whose hole is '
                         'VISIBLE from --view: some masks are a whole eyeball (a '
                         'disconnected component of the COMA template), which shows '
                         'nothing from outside and renders as an all-blue error map.')
    ap.add_argument('--test_dir', default=TEST_DIR)
    ap.add_argument('--gt_dir', default=None)
    ap.add_argument('--mask_dir', default=None)
    ap.add_argument('--out_dir', default=None,
                    help='default: <test_dir>/paper_figures')
    ap.add_argument('--clim', type=float, default=None,
                    help='max of the shared error scale, in MM (default: p99)')
    ap.add_argument('--cmap', default='jet',
                    help="jet matches the existing figures; 'turbo' or 'magma' "
                         "are perceptually better if you are willing to re-render all")
    ap.add_argument('--cbar_frac', type=float, default=0.62,
                    help='colorbar height as a fraction of the panel block '
                         '(matches the qualitative figure; a full-height bar '
                         'reads as an extra column and spreads the ticks out)')
    ap.add_argument('--view', default='xy')
    ap.add_argument('--zoom', type=float, default=1.45)
    ap.add_argument('--no_shadow', action='store_true',
                    help='disable the soft drop shadow behind each mesh')
    ap.add_argument('--shadow_blur', type=float, default=9.0)
    ap.add_argument('--shadow_strength', type=float, default=0.42,
                    help='0 = invisible, 1 = black. Keep it subtle in print.')
    ap.add_argument('--shadow_offset', type=int, nargs=2, default=(14, 10),
                    metavar=('DOWN', 'RIGHT'))
    ap.add_argument('--width_in', type=float, default=7.16,
                    help='figure width in inches (7.16 = CVPR two-column span)')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                    help='output extensions, e.g. --formats jpg')
    args = ap.parse_args()

    gt_dir = args.gt_dir or os.path.join(args.test_dir, 'gt')
    mask_dir = args.mask_dir or os.path.join(args.test_dir, 'partial')
    out_dir = args.out_dir or os.path.join(args.test_dir, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    samples = [load_sample(n, gt_dir, mask_dir) for n in args.samples]

    # ---- load every completion, compute errors in mm -----------------------
    for s in samples:
        s['comp'], s['err'], s['mm'] = [], [], []
        for label, folder in ARMS:
            path = os.path.join(args.test_dir, folder, f"{s['name']}{SUFFIX}")
            if not os.path.exists(path):
                raise SystemExit(f'missing completion: {path}')
            v = np.asarray(load_mesh(path, process=False).vertices, dtype=np.float64)
            e = np.linalg.norm(v - s['gt_v'], axis=1) * 1000.0     # m -> mm
            s['comp'].append(v)
            s['err'].append(e)
            s['mm'].append(float(e[s['mask']].mean()))

    # ---- ONE shared color scale over every error panel ---------------------
    all_err = np.concatenate([e for s in samples for e in s['err']])
    vmax = args.clim if args.clim is not None else float(np.percentile(all_err, 99))
    clim = (0.0, vmax)
    print(f'shared error scale: 0 - {vmax:.2f} mm  (cmap={args.cmap})')

    # ---- render every panel ------------------------------------------------
    ncol = 2 + len(ARMS)
    for s in samples:
        print(f"  rendering {s['name']} ...")
        shaded = [render(s['gt_v'], s['partial_faces'], zoom=args.zoom, view=args.view),
                  render(s['gt_v'], s['faces'], zoom=args.zoom, view=args.view)]
        shaded += [render(v, s['faces'], zoom=args.zoom, view=args.view)
                   for v in s['comp']]
        err = [render(v, s['faces'], scalars=e, clim=clim, cmap=args.cmap,
                      zoom=args.zoom, view=args.view)
               for v, e in zip(s['comp'], s['err'])]
        panels = shaded + err
        if args.no_shadow:
            panels = [p[..., :3] if p.shape[2] == 4 else p for p in panels]
        else:
            panels = [drop_shadow(p, offset=tuple(args.shadow_offset),
                                  blur=args.shadow_blur,
                                  strength=args.shadow_strength)
                      for p in panels]
        s['panels'] = panels

    # ONE crop box for the whole figure: every panel then has identical pixel
    # dimensions, so matplotlib never letterboxes and the head scale is the
    # same in every cell -- including across meshes.
    flat = union_crop([p for s in samples for p in s['panels']])
    k = len(samples[0]['panels'])
    for i, s in enumerate(samples):
        s['panels'] = flat[i * k:(i + 1) * k]

    # ---- compose -----------------------------------------------------------
    ph, pw = samples[0]['panels'][0].shape[:2]
    n = len(samples)
    panel_w = args.width_in / (ncol + 0.30)        # 0.30 -> left label gutter
    panel_h = panel_w * ph / pw

    # A thin spacer row under each mesh group holds that group's "x.xx mm"
    # labels. Without it the labels are painted over by the next group's axes.
    LBL = 0.20
    height_ratios, row_of = [], []                 # row_of[i] = (shaded, error)
    for si in range(n):
        row_of.append((len(height_ratios), len(height_ratios) + 1))
        height_ratios += [1.0, 1.0, LBL]
    nrow = len(height_ratios)
    fig_h = (2 * n + n * LBL) * panel_h + 0.24     # + column titles

    # Colorbar runs vertically alongside the panels, so it costs width (which
    # the two empty cells already waste) instead of height.
    fig = plt.figure(figsize=(args.width_in, fig_h))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, height_ratios=height_ratios,
                           left=0.034, right=0.935, top=0.945, bottom=0.012,
                           wspace=0.02, hspace=0.0)

    col_titles = ['Input', 'Ground truth'] + [a[0] for a in ARMS]

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    for si, s in enumerate(samples):
        shaded, err = s['panels'][:ncol], s['panels'][ncol:]
        r_top, r_bot = row_of[si]

        for c in range(ncol):
            ax = bare(fig.add_subplot(gs[r_top, c]))
            ax.imshow(shaded[c])
            if si == 0:
                # uniform weight: no column is singled out as "the" method
                ax.set_title(col_titles[c], fontsize=8, pad=3)

        # sample caption sits in the two empty cells of the error row
        cap = bare(fig.add_subplot(gs[r_bot, 0:2]))
        cap.axis('off')
        cap.text(0.5, 0.5, f"{100 * s['mask'].mean():.0f}% missing",
                 ha='center', va='center', fontsize=7.6,
                 transform=cap.transAxes, color='0.2')

        best_k = int(np.argmin(s['mm']))
        for k in range(len(ARMS)):
            ax = bare(fig.add_subplot(gs[r_bot, 2 + k]))
            ax.imshow(err[k])
            # Figure-level text draws above every axes patch, so it survives
            # sitting in the gap below the panel.
            p = ax.get_position()
            fig.text(p.x0 + p.width / 2, p.y0 - 0.004,
                     f"{s['mm'][k]:.2f} mm", ha='center', va='top',
                     fontsize=7.4, weight='bold' if k == best_k else 'normal',
                     color='#0b6623' if k == best_k else '0.15')

        # row labels in the left gutter, aligned to each row's centre.
        # Read the cell geometry off the SubplotSpec -- calling add_subplot here
        # would stack a blank axes on top of the panels already drawn.
        for r, lab in ((r_top, 'shaded'), (r_bot, 'error')):
            pos = gs[r, 0].get_position(fig)
            fig.text(0.011, pos.y0 + pos.height / 2, lab, rotation=90,
                     ha='center', va='center', fontsize=6.8, color='0.45')

    # Shared vertical colorbar, centred on the panel block. Spanning the full
    # height makes it read as an extra column and spreads the ticks too far.
    y_top = gs[row_of[0][0], 0].get_position(fig).y1
    y_bot = gs[row_of[-1][1], 0].get_position(fig).y0
    h = (y_top - y_bot) * args.cbar_frac
    cax = fig.add_axes([0.945, (y_top + y_bot) / 2 - h / 2, 0.013, h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim),
                                        cmap=args.cmap),
                      cax=cax, orientation='vertical')
    cb.set_label('per-vertex error (mm)', fontsize=7.0, labelpad=3, rotation=90)
    cb.ax.tick_params(labelsize=6.4, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)

    for ext in args.formats:
        out = os.path.join(out_dir, f'ladder_figure.{ext}')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        print('wrote', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
