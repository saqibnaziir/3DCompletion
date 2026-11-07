"""
Render every mesh asset for the paper's two schematic figures from ONE mask.

Supersedes the data path in render_architecture_assets.py, which (a) defaults to
test_008 -- a mask that is exactly the disconnected eyeball component, so its
geodesic depth field is empty and the ramp degenerates to a flat w -- and (b)
still expects the pre-reorganization flat test_set layout, where masks sat
beside the meshes rather than in test_set/partial/.

Everything here is read from disk. The completions were dumped by the full-method
arm (predict_x0_depthw_nrm_scratch) and the K-sample stack by the uncertainty
benchmark, so no checkpoint is loaded and no sampling is run.

One sample, one mask, one camera, for both figures:

  Fig. 1 (pipeline)      face_partial, face_mask, diff_0..4,
                         out_completed, div_0..2, out_uncertainty
  Fig. 2 (objective)     ramp_weights, ramp_bar, nrm_error, nrm_bar, nrm_quiver

Run (3dmesh env):
  python research/render_figure_assets.py
  python research/render_figure_assets.py --sample test_004     # plane-cut variant
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import trimesh
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.boundary_field import BoundaryFieldSolver
from utils.diffusion import GaussianDiffusion

# ---- style, shared with the draw.io palette (Academic Indigo) ---------------
MESH_COLOR = '#d9dbe3'
KNOWN_COLOR = '#b9c6e0'
MISSING_COLOR = '#d24b4b'
GT_ARROW = '#8b91a1'
PRED_ARROW = '#4f46e5'
WINDOW = (820, 940)
PATCH_WINDOW = (820, 820)
PATCH_YAW_DEG = 40.0      # swung toward whichever side the hole is on

# Sequential ramps. Both start pale so the shallow end sits quietly next to the
# grey known surface and the eye is drawn to the deep / high-error end.
CMAP_RAMP = LinearSegmentedColormap.from_list(
    'ramp', ['#dfe3fa', '#a9b4ee', '#4f46e5', '#2b1f8f'])
CMAP_ERR = LinearSegmentedColormap.from_list(
    'nrmerr', ['#f1f3f8', '#f6c453', '#e8703a', '#b91c1c'])


class Cam:
    """Fixed camera computed once from the template, reused for every snapshot.

    `yaw` swings the eye around the vertical axis. The body of the figure uses
    the frontal view (yaw=0) so all panels register vertex-for-vertex; the
    close-ups take a yaw toward the hole, which sits on the cheek and neck and
    is badly foreshortened head-on.
    """

    def __init__(self, verts, zoom=1.0, center=None, yaw=0.0):
        ext = verts.max(0) - verts.min(0)
        self.center = verts.mean(0) if center is None else np.asarray(center)
        self.dist = float(np.linalg.norm(ext)) * 2.5
        self.pscale = float(ext[1]) * 0.62 / zoom
        self.yaw = float(np.radians(yaw))

    def eye(self):
        return np.array([np.sin(self.yaw), 0.0, np.cos(self.yaw)])

    def apply(self, p):
        p.camera_position = 'xy'
        p.camera.focal_point = tuple(self.center)
        p.camera.position = tuple(self.center + self.eye() * self.dist)
        p.camera.up = (0.0, 1.0, 0.0)
        p.camera.parallel_scale = self.pscale


def make_plotter(window=WINDOW):
    p = pv.Plotter(off_screen=True, window_size=list(window))
    p.enable_parallel_projection()
    try:
        p.enable_anti_aliasing('ssaa')
    except Exception:
        pass
    return p


def pv_faces(faces):
    return np.column_stack((np.full(len(faces), 3), faces)).ravel()


def snap(cam, verts, faces, path, scalars=None, cmap=None, clim=None,
         edges=False, window=WINDOW, color=MESH_COLOR):
    p = make_plotter(window)
    m = pv.PolyData(verts, pv_faces(faces))
    if scalars is not None:
        m.point_data['s'] = scalars
        p.add_mesh(m, scalars='s', cmap=cmap, clim=clim, show_scalar_bar=False,
                   smooth_shading=True, specular=0.15, specular_power=12)
    else:
        p.add_mesh(m, color=color, smooth_shading=not edges, show_edges=edges,
                   edge_color='#9aa1b8', line_width=0.4,
                   specular=0.15, specular_power=12)
    cam.apply(p)
    p.screenshot(path, transparent_background=True)
    p.close()
    print('  ->', os.path.basename(path))


def snap_split(cam, verts, faces, mask, path, scalars, cmap, clim,
               window=WINDOW):
    """Grey known surface, scalar field only inside the hole.

    Colouring the whole mesh puts the known region at the bottom of the ramp,
    where it is nearly indistinguishable from the neutral grey used in every
    other panel -- the hole stops reading as the subject. Splitting the face
    list gives a crisp seam for free and keeps the colour budget on the region
    the term actually acts on.
    """
    touches = np.any(mask[faces], axis=1)
    p = make_plotter(window)
    p.add_mesh(pv.PolyData(verts, pv_faces(faces[~touches])), color=MESH_COLOR,
               smooth_shading=True, specular=0.15, specular_power=12)
    hole = pv.PolyData(verts, pv_faces(faces[touches]))
    hole.point_data['s'] = scalars
    p.add_mesh(hole, scalars='s', cmap=cmap, clim=clim, show_scalar_bar=False,
               smooth_shading=True, specular=0.15, specular_power=12)
    cam.apply(p)
    p.screenshot(path, transparent_background=True)
    p.close()
    print('  ->', os.path.basename(path))


def colorbar(path, cmap, clim, label, ticks=None):
    """Standalone horizontal colorbar; draw.io places it as its own image."""
    # The bar itself is a thin strip; most of the box is reserved for the tick
    # row and the label, which get clipped if the figure is sized to the strip.
    fig, ax = plt.subplots(figsize=(2.6, 0.74))
    fig.subplots_adjust(bottom=0.52, top=0.80, left=0.06, right=0.94)
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        norm=norm, orientation='horizontal')
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:g}' if float(t).is_integer() else f'{t:.1f}'
                           for t in ticks])
    cb.set_label(label, fontsize=7, color='#1f2937')
    cb.ax.tick_params(labelsize=6, colors='#1f2937', length=2)
    cb.outline.set_edgecolor('#cdd2dc')
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    print('  ->', os.path.basename(path))


def vertex_normals(verts, faces):
    """Area-weighted unit vertex normals -- same quantity as utils/normal_loss."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)          # unnormalized => area-weighted
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='test_010',
                    help='hero sample; must have a connected hole (test_008 and '
                         'test_000 are the disconnected eyeball -- no depth field)')
    ap.add_argument('--test_dir',
                    default=os.path.join('data', 'completion_test_samples', 'test_set'))
    ap.add_argument('--arm', default='completed_diffusion_x0_depthw_nrm_scratch',
                    help='full-method completions, used for every mesh render')
    ap.add_argument('--motive_arm', default='completed_diffusion_x0_depthw',
                    help='arm WITHOUT the normal term; the decoupling that '
                         'motivates the term must be read off this one')
    ap.add_argument('--scale', default=os.path.join(
        'data', 'checkpoints_diffusion', 'predict_x0_depthw_nrm_scratch',
        'diffusion_scale.pt'))
    ap.add_argument('--template', default=os.path.join('data', 'template.ply'))
    ap.add_argument('--out', default=os.path.join('docs', 'Model_archi', 'assets'))
    ap.add_argument('--w_max', type=float, default=3.0)
    ap.add_argument('--timesteps', type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    td, s = args.test_dir, args.sample

    # ---- load: template, gt, mask, completion, K-stack ----------------------
    tpl = trimesh.load(args.template, process=False)
    T = np.asarray(tpl.vertices, dtype=np.float64)
    F = np.asarray(tpl.faces, dtype=np.int64)

    gt = np.asarray(trimesh.load(
        os.path.join(td, 'gt', f'{s}_gt_complete.ply'), process=False).vertices,
        dtype=np.float64)
    mask = np.load(os.path.join(td, 'partial', f'{s}_mask.npy')).astype(bool)
    pred = np.asarray(trimesh.load(
        os.path.join(td, args.arm, f'{s}_completed_diffusion.ply'),
        process=False).vertices, dtype=np.float64)

    sigma = torch.load(args.scale, map_location='cpu',
                       weights_only=False)['sigma'].numpy().astype(np.float64)

    cam = Cam(T)
    # Swing the close-up camera toward the side the hole is on, so the region
    # under discussion is not seen edge-on.
    hole_ctr = gt[mask].mean(0)
    patch_yaw = PATCH_YAW_DEG * (1.0 if hole_ctr[0] >= T.mean(0)[0] else -1.0)
    print(f'sample={s}  missing={int(mask.sum())} ({mask.mean()*100:.1f}%)  '
          f'patch yaw={patch_yaw:+.0f} deg')

    # ---- Fig.1: partial + mask ---------------------------------------------
    keep = F[~np.any(mask[F], axis=1)]
    snap(cam, gt, keep, os.path.join(args.out, 'face_partial.png'))
    snap(cam, gt, F, os.path.join(args.out, 'face_mask.png'),
         scalars=mask.astype(float), cmap=[KNOWN_COLOR, MISSING_COLOR], clim=[0, 1])

    # ---- Fig.1: forward diffusion, the REAL q_sample ------------------------
    # x0 = Delta/sigma is what the model sees; we noise there and map back to
    # coordinates (T + sigma*x_t) purely for display, so the strip shows the
    # actual schedule rather than an ad-hoc vertex jitter.
    diff = GaussianDiffusion(timesteps=args.timesteps, schedule='cosine')
    x0 = (gt - T) / sigma
    rng = np.random.RandomState(0)
    eps = rng.randn(*x0.shape)
    ts = [0, 250, 500, 750, args.timesteps - 1]
    acp = diff.alphas_cumprod.numpy()
    for i, t in enumerate(ts):
        xt = np.sqrt(acp[t]) * x0 + np.sqrt(1.0 - acp[t]) * eps
        snap(cam, T + sigma * xt, F, os.path.join(args.out, f'diff_{i}.png'),
             edges=True)
    print(f'  q_sample at t={ts}  (alpha_bar={[round(float(acp[t]),3) for t in ts]})')

    # ---- Fig.1: completion, diversity, uncertainty --------------------------
    snap(cam, pred, F, os.path.join(args.out, 'out_completed.png'))

    hero = os.path.join(td, 'uncertainty_main', 'hero_samples.npz')
    stack = None
    if os.path.exists(hero):
        z = np.load(hero, allow_pickle=True)
        if f'{s}_stack' in z.files:
            stack = z[f'{s}_stack']
    if stack is None:
        print(f'  ! no K-sample stack for {s}; skipping diversity + uncertainty')
    else:
        K = len(stack)
        for k in range(3):
            snap(cam, stack[k], F, os.path.join(args.out, f'div_{k}.png'))
        unc = np.linalg.norm(stack.std(0), axis=1) * 1000.0        # mm
        hi = float(np.percentile(unc[mask], 95))
        snap(cam, stack[0], F, os.path.join(args.out, 'out_uncertainty.png'),
             scalars=unc, cmap='turbo', clim=[0.0, hi])
        colorbar(os.path.join(args.out, 'unc_bar.png'), 'turbo', [0.0, hi],
                 f'per-vertex std over K={K} (mm)')
        print(f'  uncertainty: K={K}  known max={unc[~mask].max():.3f} mm  '
              f'missing p95={hi:.2f} mm')

    # ---- Fig.2: geodesic depth ramp ----------------------------------------
    solver = BoundaryFieldSolver(F, T, backend='dijkstra')
    d, ring = solver.depth(mask)
    if len(ring) == 0:
        raise SystemExit(
            f'{s} has an empty boundary ring (hole is a disconnected component), '
            'so there is no depth field to draw. Pick a sample with a connected '
            'hole -- see the --sample help.')
    w = solver.depth_weight(mask, w_max=args.w_max, normalize=True, d=d)

    # normalize=True rescales the ramp so its MEAN over the hole is w_max, which
    # pushes the deepest weights ABOVE w_max -- the bar must span the real range.
    w_hi = float(w[mask].max())
    snap_split(cam, gt, F, mask, os.path.join(args.out, 'ramp_weights.png'),
               scalars=w, cmap=CMAP_RAMP, clim=[1.0, w_hi])
    cam_ramp = Cam(T, zoom=1.9, center=hole_ctr, yaw=patch_yaw)
    snap_split(cam_ramp, gt, F, mask,
               os.path.join(args.out, 'ramp_weights_zoom.png'),
               scalars=w, cmap=CMAP_RAMP, clim=[1.0, w_hi], window=PATCH_WINDOW)
    colorbar(os.path.join(args.out, 'ramp_bar.png'), CMAP_RAMP, [1.0, w_hi],
             r'loss weight $w_i$', ticks=[1.0, w_hi])
    print(f'  ramp: ring={len(ring)}  d_max={d[mask][np.isfinite(d[mask])].max()*1000:.1f} mm  '
          f'w={w[mask].min():.2f}..{w_hi:.2f}  mean={w[mask].mean():.2f}')

    # ---- Fig.2: normal term -------------------------------------------------
    # Normals on UN-STANDARDIZED coordinates, exactly as the loss computes them.
    n_gt = vertex_normals(gt, F)
    n_pr = vertex_normals(pred, F)
    cos = np.clip((n_gt * n_pr).sum(1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    hi = float(np.percentile(ang[mask], 95))
    print(f'  normals: mean angle over hole={ang[mask].mean():.2f} deg  p95={hi:.2f} deg')

    cam_patch = Cam(T, zoom=1.9, center=hole_ctr, yaw=patch_yaw)
    snap_split(cam_patch, pred, F, mask, os.path.join(args.out, 'nrm_error.png'),
               scalars=ang, cmap=CMAP_ERR, clim=[0.0, hi], window=PATCH_WINDOW)
    colorbar(os.path.join(args.out, 'nrm_bar.png'), CMAP_ERR, [0.0, hi],
             r'normal error $\angle(\hat{n},n)$ (deg)')

    # A quiver of predicted vs ground-truth normals was tried here and dropped:
    # at 4.5 deg mean error the two arrow sets sit on top of each other, so the
    # panel showed nothing. What the normal term needs to justify itself is the
    # decoupling measurement, not a picture of two overlapping arrows.
    rho_strip(td, F, args.motive_arm,
              os.path.join(args.out, 'rho_strip.png'), highlight=s)

    print('\nDone. Assets in', args.out)


def rho_strip(test_dir, faces, arm, path, highlight=None):
    """
    Per-completion Spearman correlation between positional and normal error.

    One dot per test completion, measured WITHIN that completion. This is the
    quantity the paper's decoupling argument rests on, and showing the whole
    distribution rather than a single averaged number also shows its spread --
    which pooling every vertex across completions would hide behind the
    between-sample difficulty confound.

    Measured on the arm WITHOUT the normal term: the claim motivates adding it,
    so it has to be read off the model as it was before.
    """
    from glob import glob
    from scipy.stats import spearmanr

    rows = []
    for f in sorted(glob(os.path.join(test_dir, arm, '*_completed_diffusion.ply'))):
        s = os.path.basename(f).replace('_completed_diffusion.ply', '')
        gt_p = os.path.join(test_dir, 'gt', f'{s}_gt_complete.ply')
        mk_p = os.path.join(test_dir, 'partial', f'{s}_mask.npy')
        if not (os.path.exists(gt_p) and os.path.exists(mk_p)):
            continue
        gt = np.asarray(trimesh.load(gt_p, process=False).vertices, dtype=np.float64)
        pr = np.asarray(trimesh.load(f, process=False).vertices, dtype=np.float64)
        m = np.load(mk_p).astype(bool)
        if m.sum() < 10:
            continue
        pe = np.linalg.norm(pr - gt, axis=1)[m]
        an = np.degrees(np.arccos(np.clip(
            (vertex_normals(gt, faces) * vertex_normals(pr, faces)).sum(1),
            -1.0, 1.0)))[m]
        rows.append((s, float(spearmanr(pe, an).statistic)))

    names = [r[0] for r in rows]
    rho = np.array([r[1] for r in rows])
    med = float(np.median(rho))

    fig, ax = plt.subplots(figsize=(3.4, 1.05))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.72, bottom=0.42)
    rng = np.random.RandomState(0)
    y = rng.uniform(-0.32, 0.32, len(rho))
    ax.axvline(0.0, color='#9aa1b8', lw=0.8, zorder=1)
    ax.scatter(rho, y, s=26, facecolor=PRED_ARROW, edgecolor='white',
               linewidth=0.6, alpha=0.9, zorder=3)
    if highlight in names:
        i = names.index(highlight)
        ax.scatter([rho[i]], [y[i]], s=64, facecolor='none',
                   edgecolor='#b91c1c', linewidth=1.3, zorder=4)
    ax.axvline(med, color='#2b1f8f', lw=1.4, zorder=2)
    ax.text(med, 0.62, f'median {med:+.2f}', ha='center', va='bottom',
            fontsize=7, color='#2b1f8f')
    ax.set_ylim(-0.75, 0.95)
    ax.set_yticks([])
    ax.set_xlabel(r'Spearman $\rho$ (positional vs. normal error), per completion',
                  fontsize=7, color='#1f2937')
    ax.tick_params(labelsize=6.5, colors='#1f2937', length=2)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#cdd2dc')
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    print('  ->', os.path.basename(path),
          f'(n={len(rho)}  median={med:+.3f}  mean={rho.mean():+.3f}  '
          f'range {rho.min():+.2f}..{rho.max():+.2f})')


if __name__ == '__main__':
    main()
