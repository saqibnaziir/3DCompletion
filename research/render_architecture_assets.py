"""
Render REAL mesh snapshots for the model-architecture figure (no fake AI faces).

Everything is rendered from your own data / trained model with one fixed camera,
fixed lighting, and a transparent background, so the assets are mutually
consistent and drop cleanly onto any panel color.

Outputs (docs/figures/assets/, transparent PNG):
  face_partial.png      real partial mesh (faces touching the hole removed)
  face_mask.png         same mesh, vertices colored Known (light) / Missing (red)
  diff_0..diff_4.png    forward-diffusion of the offset field at t=0..T
                        (real q_sample, shown in the standardized space the model
                         operates in: t=0 = smooth offsets, t=T = white noise)
  level_5023/2511/1255/627.png   the 4 Graph-U-Net resolutions (template down-
                        sampled by the real D matrices), drawn as graph nodes
  out_completed.png     your model's completion (sample 0)
  div_0..div_2.png      diverse completions (different noise seeds)
  out_uncertainty.png   per-vertex std over K samples (heatmap; ~0 on known region)

Run (in the 3dmesh env, GPU):
  python research/render_architecture_assets.py --config_file research/configs/config_diffusion.cfg --sample test_008
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import pyvista as pv
from glob import glob
from trimesh import load_mesh

from config.config import read_config
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     complete, get_template_info, load_norm)

# ---- consistent style -------------------------------------------------------
MESH_COLOR = '#d9dbe3'      # neutral light gray surface
NODE_COLOR = '#3b3f5c'      # slate graph nodes
KNOWN_COLOR = '#b9c6e0'     # light indigo  (known / observed)
MISSING_COLOR = '#d24b4b'   # muted red     (missing)
WINDOW = (820, 940)


def make_plotter():
    p = pv.Plotter(off_screen=True, window_size=list(WINDOW))
    p.enable_parallel_projection()
    try:
        p.enable_anti_aliasing('ssaa')
    except Exception:
        pass
    return p


class Cam:
    """Fixed camera computed once from the template, reused for every snapshot."""
    def __init__(self, verts):
        c = verts.mean(0)
        ext = verts.max(0) - verts.min(0)
        self.center = c
        self.dist = float(np.linalg.norm(ext)) * 2.5
        self.pscale = float(ext[1]) * 0.62          # frame to face height

    def apply(self, p):
        p.camera_position = 'xy'
        p.camera.focal_point = tuple(self.center)
        p.camera.position = (self.center[0], self.center[1], self.center[2] + self.dist)
        p.camera.up = (0.0, 1.0, 0.0)
        p.camera.parallel_scale = self.pscale


def pv_faces(faces):
    return np.column_stack((np.full(len(faces), 3), faces)).ravel()


def snap_surface(cam, verts, faces, path, scalars=None, cmap=None, clim=None, edges=False):
    p = make_plotter()
    m = pv.PolyData(verts, pv_faces(faces))
    if scalars is not None:
        m.point_data['s'] = scalars
        p.add_mesh(m, scalars='s', cmap=cmap, clim=clim, show_scalar_bar=False,
                   smooth_shading=True, specular=0.15, specular_power=12)
    else:
        p.add_mesh(m, color=MESH_COLOR, smooth_shading=not edges,
                   show_edges=edges, edge_color='#9aa1b8', line_width=0.4,
                   specular=0.15, specular_power=12)
    cam.apply(p)
    p.screenshot(path, transparent_background=True)
    p.close()
    print('  ->', path)


def snap_points(cam, verts, path, point_size):
    p = make_plotter()
    p.add_mesh(pv.PolyData(verts), color=NODE_COLOR, render_points_as_spheres=True,
               point_size=point_size)
    cam.apply(p)
    p.screenshot(path, transparent_background=True)
    p.close()
    print('  ->', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--sample', default='test_008', help='test basename to feature')
    ap.add_argument('--test_dir', default=None)
    ap.add_argument('--out', default=os.path.join('docs', 'figures', 'assets'))
    ap.add_argument('--num_samples', type=int, default=5)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--noise_scale', type=float, default=0.05,
                    help='max vertex-noise amplitude on the diffusion mesh strip, '
                         'as a fraction of face height (bigger = rougher final mesh)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples', 'test_set')

    # template + faces + camera
    template_mesh, _ = get_template_info(config)
    tmpl = np.asarray(template_mesh.vertices, dtype=np.float64)
    faces = np.asarray(template_mesh.faces)
    cam = Cam(tmpl)

    # the featured sample
    gt = np.asarray(load_mesh(os.path.join(test_dir, f'{args.sample}_gt_complete.ply'),
                              process=False).vertices, dtype=np.float64)
    mask = np.load(os.path.join(test_dir, f'{args.sample}_mask.npy')).astype(bool)

    # ---- 1. partial (drop faces touching the hole) ----
    keep = faces[~np.any(mask[faces], axis=1)]
    snap_surface(cam, gt, keep, os.path.join(args.out, 'face_partial.png'))

    # ---- 2. known / missing mask ----
    snap_surface(cam, gt, faces, os.path.join(args.out, 'face_mask.png'),
                 scalars=mask.astype(float), cmap=[KNOWN_COLOR, MISSING_COLOR], clim=[0, 1])

    # ---- 3. forward diffusion, visualized as NOISE ON THE MESH (faces kept):
    #         x_t = x0 + amp_t * noise, amplitude rising across the strip so the
    #         face stays a recognizable surface but gets progressively rougher
    #         (t=0 = clean mesh, last frame = heavily noised mesh). ----
    sigma = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location='cpu', weights_only=False)['sigma'].numpy()
    ext_y = float(tmpl.max(0)[1] - tmpl.min(0)[1])
    amps = np.array([0.0, 0.22, 0.46, 0.72, 1.0]) * args.noise_scale * ext_y
    np.random.seed(0)
    for i, amp in enumerate(amps):
        xt = gt + np.random.randn(*gt.shape) * amp        # displace vertices, keep faces
        snap_surface(cam, xt, faces, os.path.join(args.out, f'diff_{i}.png'), edges=True)

    # ---- 4. Graph U-Net resolutions as REAL meshes (vertices + faces per level) ----
    from utils.funcs import generate_VFDU
    Vlist, Flist, _, _ = generate_VFDU(template_mesh, config['down_sampling_factors'])
    counts = []
    for v, f in zip(Vlist, Flist):
        v = v.numpy() if hasattr(v, 'numpy') else np.asarray(v, dtype=np.float64)
        f = np.asarray(f)
        counts.append(len(v))
        snap_surface(cam, v, f, os.path.join(args.out, f'level_{len(v)}.png'), edges=True)
    print('  U-Net levels:', counts)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'], schedule=dcfg['schedule'])

    # ---- 5/6/7. completion, diverse samples, uncertainty ----
    denoiser = build_denoiser(config, dcfg, device, use_ema=True)
    clip_value = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                            map_location='cpu', weights_only=False).get('clip_value', 6.0)
    sig_t = torch.tensor(sigma, device=device)
    partial = gt.copy().astype(np.float32); partial[mask] = 0.0
    samples = complete(denoiser, diffusion.to(device), partial, mask, config, device,
                       sig_t, clip_value=clip_value, num_steps=args.steps,
                       num_samples=args.num_samples)
    snap_surface(cam, samples[0], faces, os.path.join(args.out, 'out_completed.png'))
    for k in range(min(3, len(samples))):
        snap_surface(cam, samples[k], faces, os.path.join(args.out, f'div_{k}.png'))
    stack = np.stack(samples, 0)
    unc = np.linalg.norm(stack.std(0), axis=1)
    clim = [0.0, float(np.percentile(unc[mask], 95)) if mask.any() else float(unc.max())]
    snap_surface(cam, samples[0], faces, os.path.join(args.out, 'out_uncertainty.png'),
                 scalars=unc, cmap='turbo', clim=clim)

    print('\nDone. Assets in', args.out)


if __name__ == '__main__':
    main()
