"""
Evaluate a trained diffusion arm on the fixed official-COMA-test benchmark
(built by build_test_benchmark.py).

Per sample: missing-region / overall per-vertex L2 (mm), and vertex-normal
angular error (deg) on the missing region and on the 2-ring seam band.
Writes one CSV row per sample as it goes, so an interrupted run resumes where
it stopped. Prints a summary by mask type and hole size at the end.

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/eval_official_test.py --param x0 --tag depthw_nrm
Options:
    --use_last        last-epoch EMA weights instead of best-validation EMA
    --save_meshes K   also save the first K completed meshes (default 0)
    --limit N         only the first N samples (smoke test)
"""
import os
import sys
import csv
import argparse

import numpy as np
import scipy.sparse as sp
import torch
from trimesh import Trimesh, load_mesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from config.config import read_config
from utils.diffusion import GaussianDiffusion
from utils.funcs import save_ply_explicit
from utils.normals import normal_angular_error, boundary_band
from scripts.infer_diffusion import (parse_diffusion_section, build_denoiser,
                                     get_template_info, complete)

# Half-open bins covering everything (a few masks land just outside 5-40%).
SIZE_BINS = [(0.0, 0.15), (0.15, 0.25), (0.25, 1.01)]
FIELDS = ['row', 'subject', 'expression', 'frame', 'strategy', 'missing_pct',
          'missing_mm', 'overall_mm', 'nrm_missing_deg', 'nrm_seam_deg']


def adjacency(faces, n):
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                        faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                        faces[:, 0], faces[:, 1], faces[:, 2]])
    a = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def summarize(rows):
    def line(name, sel):
        if not sel:
            return
        m = np.array([r['missing_mm'] for r in sel])
        nm = np.array([r['nrm_missing_deg'] for r in sel])
        ns = np.array([r['nrm_seam_deg'] for r in sel])
        print(f'  {name:22s} n={len(sel):5d}  missing {m.mean():.3f} +/- {m.std():.3f} mm'
              f'  | normals hole {np.nanmean(nm):.2f} deg, seam {np.nanmean(ns):.2f} deg')

    print('\n' + '=' * 92)
    line('ALL', rows)
    for s in ('random_scatter', 'bfs_patch', 'plane_cut'):
        line(s, [r for r in rows if r['strategy'] == s])
    for lo, hi in SIZE_BINS:
        line(f'hole {int(lo*100)}-{min(int(hi*100), 100)}%',
             [r for r in rows if lo <= r['missing_pct'] / 100 < hi])
    print('=' * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file',
                    default=os.path.join(_ROOT, 'config', 'config_diffusion_revision.cfg'))
    ap.add_argument('--param', required=True, choices=['eps', 'x0', 'v'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--bench_dir', default=os.path.join(_ROOT, 'revision', 'benchmark'))
    ap.add_argument('--out_dir', default=None,
                    help='default: revision/results/official_test/<arm>[_last]')
    ap.add_argument('--use_last', action='store_true')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_meshes', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    dcfg['predict'] = args.param
    arm = f'predict_{args.param}' + (f'_{args.tag}' if args.tag else '')
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    num_steps = args.steps or dcfg['sampling_steps']
    out_dir = args.out_dir or os.path.join(_ROOT, 'revision', 'results', 'official_test',
                                           arm + ('_last' if args.use_last else ''))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    denoiser = build_denoiser(config, dcfg, device, use_ema=True, use_last=args.use_last)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'], schedule=dcfg['schedule'],
                                  predict=dcfg['predict']).to(device)
    template_mesh, _ = get_template_info(config)
    faces = np.asarray(template_mesh.faces)
    n_verts = len(template_mesh.vertices)
    adj = adjacency(faces, n_verts)

    scale = torch.load(os.path.join(config['checkpoint_dir'], 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma, clip_value = scale['sigma'].to(device), scale.get('clip_value')

    masks = np.load(os.path.join(args.bench_dir, 'official_test_masks.npz'))['masks']
    with open(os.path.join(args.bench_dir, 'official_test_index.csv')) as fh:
        index = list(csv.DictReader(fh))
    if args.limit:
        index = index[:args.limit]

    csv_path = os.path.join(out_dir, 'per_sample.csv')
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                done[int(r['row'])] = {k: (r[k] if k in ('subject', 'expression', 'frame',
                                                          'strategy') else float(r[k]))
                                       for k in FIELDS}
    print(f'arm={arm}  weights={"last" if args.use_last else "best-val"}  steps={num_steps}')
    print(f'{len(index)} samples, {len(done)} already done -> {csv_path}')

    new_file = not os.path.exists(csv_path)
    fh = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    gt_cache = {}
    rows = [done[k] for k in sorted(done)]
    todo = [r for r in index if int(r['row']) not in done]
    for i, rec in enumerate(todo):
        k = int(rec['row'])
        mask = masks[k]
        if rec['gt_path'] not in gt_cache:
            gt_cache.clear()
            gt_cache[rec['gt_path']] = np.asarray(
                load_mesh(os.path.join(_ROOT, rec['gt_path']), process=False).vertices,
                dtype=np.float32)
        gt = gt_cache[rec['gt_path']]

        partial = gt.copy()
        partial[mask] = 0.0
        comp = complete(denoiser, diffusion, partial, mask, config, device, sigma,
                        clip_value=clip_value, num_steps=num_steps, num_samples=1)[0]

        err = np.linalg.norm(comp - gt, axis=1) * 1000.0         # m -> mm
        ang, valid = normal_angular_error(comp, gt, faces)
        seam = boundary_band(adj, mask, rings=2)
        row = dict(row=k, subject=rec['subject'], expression=rec['expression'],
                   frame=rec['frame'], strategy=rec['strategy'],
                   missing_pct=round(100.0 * mask.mean(), 3),
                   missing_mm=float(err[mask].mean()), overall_mm=float(err.mean()),
                   nrm_missing_deg=float(np.nanmean(ang[mask & valid])),
                   nrm_seam_deg=float(np.nanmean(ang[seam & valid])))
        writer.writerow(row)
        fh.flush()
        rows.append(row)

        if k < args.save_meshes:
            mesh_dir = os.path.join(out_dir, 'meshes')
            os.makedirs(mesh_dir, exist_ok=True)
            name = f"{k:05d}_{rec['subject']}_{rec['expression']}.{rec['frame']}_{rec['strategy']}"
            save_ply_explicit(Trimesh(vertices=comp, faces=faces, process=False),
                              os.path.join(mesh_dir, name + '_completed.ply'))
            np.save(os.path.join(mesh_dir, name + '_mask.npy'), mask)

        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            m = np.mean([r['missing_mm'] for r in rows])
            print(f'  [{len(rows)}/{len(index)}] running missing error {m:.3f} mm', flush=True)
    fh.close()

    summarize(rows)
    print('per-sample CSV:', csv_path)


if __name__ == '__main__':
    main()
