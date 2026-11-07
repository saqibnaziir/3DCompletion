"""
Post-hoc analysis of a `complete_real_partial.py` run on the cropped Test_old
meshes: how does completion error behave as a function of DEPTH into the hole?

Reads the completed meshes that run already wrote (no resampling, no GPU), and
for each mesh bins the per-vertex error by geodesic distance from the known
boundary ring (`utils/boundary_field.BoundaryFieldSolver.depth`, the same field
the depth-weighted loss uses).

The half-missing cases are the point: if error were uniform over the hole the
model would just be interpolating badly everywhere, but if it ramps with depth
the model is anchored at the seam and extrapolating -- which is the behaviour a
completion model should show, and the one the uncertainty map has to track.

Writes depth_profile.csv + depth_profile.png next to the completed meshes.

Run (conda 3dmesh python):
  python research/analyze_real_partial.py \
      --comp_dir data/completion_test_samples/test_set/Test_old/completed_x0_depthw_nrm_scratch \
      --gt_dir   data/completion_test_samples/test_set/Test_old/gt \
      --partial_dir data/completion_test_samples/test_set/Test_old/partial
"""
import os, sys, argparse, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from trimesh import load_mesh
from scipy.spatial import cKDTree

from config.config import read_config, resolve_config_file
from utils.boundary_field import BoundaryFieldSolver
from research.complete_real_partial import gt_for
# Same definitions the COMA calibration table uses, so the numbers are comparable.
from research.eval_uncertainty_calibration import metrics_from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='COMA')
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--comp_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--partial_dir', required=True)
    ap.add_argument('--suffix', default='_completed_meanofK.ply',
                    help="which completed mesh to analyse (default: the mean-of-K "
                         "one; use _completed.ply for the first single draw)")
    ap.add_argument('--bins', type=int, default=8)
    args = ap.parse_args()

    config = read_config(resolve_config_file(args.data, args.config_file))
    tmpl = load_mesh(config['template_file'], process=False)
    faces = np.asarray(tmpl.faces, dtype=np.int64)
    verts_t = np.asarray(tmpl.vertices)
    solver = BoundaryFieldSolver(faces, verts_t)

    rows, curves, pooled = [], [], []
    for pf in sorted(glob(os.path.join(args.partial_dir, '*.ply'))):
        name = os.path.basename(pf).replace('.ply', '')
        comp_path = os.path.join(args.comp_dir, f'{name}{args.suffix}')
        gt_path = gt_for(os.path.basename(pf), args.gt_dir)
        if gt_path is None or not os.path.exists(comp_path):
            print(f'  {name}: missing gt or completed mesh, skipping'); continue

        gt = np.asarray(load_mesh(gt_path, process=False).vertices, dtype=np.float64)
        comp = np.asarray(load_mesh(comp_path, process=False).vertices, dtype=np.float64)
        part = np.asarray(load_mesh(pf, process=False).vertices, dtype=np.float64)

        d, _ = cKDTree(part).query(gt)
        missing = d > 1e-5
        if missing.sum() < 10:
            print(f'  {name}: only {missing.sum()} missing verts, skipping'); continue

        depth, _ = solver.depth(missing)              # geodesic dist from the seam
        err_mm = 1000.0 * np.linalg.norm(comp - gt, axis=1)

        dm = depth[missing]
        finite = np.isfinite(dm)
        # Unreachable vertices (the template is 3 components -- see the eyeball
        # note in utils/boundary_field.py) have infinite depth; bin them last.
        dmax = float(dm[finite].max()) if finite.any() else 0.0
        edges = np.linspace(0.0, dmax, args.bins + 1)
        em = err_mm[missing]

        prof = []
        for i in range(args.bins):
            lo, hi = edges[i], edges[i + 1]
            sel = (dm >= lo) & (dm <= hi if i == args.bins - 1 else dm < hi)
            prof.append((0.5 * (lo + hi), float(em[sel].mean()) if sel.any() else np.nan,
                         int(sel.sum())))
        curves.append((name, 100.0 * missing.mean(), prof, dmax))

        # First/last bin that actually caught vertices (tiny holes leave gaps).
        seam_bin = next((p[1] for p in prof if np.isfinite(p[1])), np.nan)
        deep_bin = next((p[1] for p in reversed(prof) if np.isfinite(p[1])), np.nan)
        row = {
            'name': name, 'missing_pct': 100.0 * missing.mean(),
            'hole_depth_max_mm': 1000.0 * dmax,
            'err_shallowest_bin_mm': seam_bin, 'err_deepest_bin_mm': deep_bin,
            'ramp_factor': deep_bin / seam_bin if seam_bin else np.nan,
            'err_mean_mm': float(em.mean()),
            'corr_depth_err': float(np.corrcoef(dm[finite], em[finite])[0, 1])
                              if finite.sum() > 2 else np.nan,
        }

        # Does the per-vertex uncertainty track the error INSIDE the hole?
        # (Outside it, both are ~0 by construction -- clamping makes known
        # vertices exact -- so scoring the whole mesh would inflate rho.)
        # `e` is the error of the MEAN of the K draws, matching
        # eval_uncertainty_calibration.py, so rho/AUSE here are comparable to
        # the COMA calibration table rather than to a differently-defined one.
        npz = os.path.join(args.comp_dir, f'{name}_pervertex.npz')
        if os.path.exists(npz):
            z = np.load(npz)
            u, e = z['unc_mm'][missing], z['err_meanK_mm'][missing]
            keep = u > 0
            if keep.sum() > 10:
                cal, _ = metrics_from(u[keep], e[keep])
                row.update(spearman_rho=cal['spearman_rho'], pearson_r=cal['pearson_r'],
                           ause=cal['ause'], aurg=cal['aurg'],
                           mean_unc_mm=cal['mean_uncertainty'])
                pooled.append((u[keep], e[keep], 100.0 * missing.mean()))
        rows.append(row)
        print(f"  {name}: depth 0..{1000*dmax:.0f} mm, err {seam_bin:.2f} -> "
              f"{deep_bin:.2f} mm ({row['ramp_factor']:.1f}x), "
              f"corr(depth,err)={row['corr_depth_err']:+.2f}"
              + (f", rho(unc,err)={row['spearman_rho']:+.3f}, AUSE={row['ause']:.3f}"
                 if 'spearman_rho' in row else ''))

    if not rows:
        raise SystemExit('nothing analysed')

    # Pooled calibration, matching the aggregate row of the COMA table (which
    # concatenates every mesh's missing vertices before scoring).
    if pooled:
        print()
        for label, sel in (('all meshes', lambda p: True),
                           ('large holes (>25% missing)', lambda p: p[2] > 25.0)):
            grp = [p for p in pooled if sel(p)]
            if not grp:
                continue
            U = np.concatenate([p[0] for p in grp])
            E = np.concatenate([p[1] for p in grp])
            cal, _ = metrics_from(U, E)
            print(f"  POOLED, {label:<28} n={cal['n_vertices']:>6}  "
                  f"rho={cal['spearman_rho']:+.3f}  r={cal['pearson_r']:+.3f}  "
                  f"AUSE={cal['ause']:.3f}  AURG={cal['aurg']:.3f}")

    csv_path = os.path.join(args.comp_dir, 'depth_profile.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, pct, prof, dmax in curves:
        x = [1000.0 * p[0] for p in prof]
        y = [p[1] for p in prof]
        big = pct > 25
        ax.plot(x, y, marker='o', lw=2.4 if big else 1.2,
                alpha=1.0 if big else 0.55,
                label=f'{name} ({pct:.0f}% missing)')
    ax.set_xlabel('geodesic depth into the hole (mm from the known boundary)')
    ax.set_ylabel('mean per-vertex error (mm)')
    ax.set_title('Completion error vs hole depth — cropped real partials')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.comp_dir, 'depth_profile.png')
    fig.savefig(png, dpi=160)
    print(f'\nwrote {csv_path}\nwrote {png}')


if __name__ == '__main__':
    main()
