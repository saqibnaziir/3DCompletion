"""
Is there headroom for a normal loss, or is normal error just a shadow of
position error?

A normal loss can only add something if part of the orientation error is
ORTHOGONAL to the positional error the current loss already sees. Three probes:

  A. rank correlation between per-vertex position error and normal error.
     Near 1.0 => normals are redundant, fixing positions fixes normals.
  B. roughness ratio: mean |L(v)| over the hole for our completion vs GT.
     >1 => we ADD high-frequency noise (a normal/smoothness term should bite)
     <1 => we OVER-SMOOTH (a normal term would have to RESTORE detail)
  C. spectral split of the error field e = comp - gt into smooth vs rough:
     |L(e)| / |e|. Compared against the same ratio for the GT surface itself,
     this says whether the error is a misplaced-but-smooth bulge (normal loss
     cannot help much) or genuine ripple (normal loss targets it directly).
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from glob import glob
from trimesh import load_mesh
from scipy.stats import spearmanr
import scipy.sparse as sp

from utils.normals import normal_angular_error, boundary_band
from utils.boundary_field import BoundaryFieldSolver

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_ROOT, 'data', 'completion_test_samples', 'test_set')
ARMS = {
    'x0':        os.path.join(BASE, 'completed_diffusion_x0'),
    'x0_depthw': os.path.join(BASE, 'completed_diffusion_x0_depthw'),
}
SUFFIX = '_completed_diffusion.ply'


def uniform_laplacian(adj, n):
    """L = I - D^-1 A  (umbrella operator; matches train_diffusion.laplacian_loss)."""
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sp.eye(n, format='csr') - sp.diags(1.0 / deg) @ adj


def main():
    gts = sorted(glob(os.path.join(BASE, 'gt', '*_gt_complete.ply')))
    tpl = load_mesh(gts[0], process=False)
    F = np.asarray(tpl.faces, dtype=np.int64)
    N = len(tpl.vertices)
    solver = BoundaryFieldSolver(F, np.asarray(tpl.vertices))
    L = uniform_laplacian(solver._A, N)

    for arm, d in ARMS.items():
        rho, rough_ratio, err_rough, gt_rough, deep_rho = [], [], [], [], []
        for g in gts:
            name = os.path.basename(g).replace('_gt_complete.ply', '')
            c = os.path.join(d, name + SUFFIX)
            m = os.path.join(BASE, 'partial', name + '_mask.npy')
            if not os.path.exists(c):
                continue
            gt = np.asarray(load_mesh(g, process=False).vertices, dtype=np.float64)
            comp = np.asarray(load_mesh(c, process=False).vertices, dtype=np.float64)
            mask = np.load(m)
            if mask.sum() < 20:
                continue

            pos = np.linalg.norm(comp - gt, axis=1)
            ang, valid = normal_angular_error(comp, gt, F)
            sel = mask & valid & (pos > 0)

            # A. are the two errors measuring the same thing?
            if sel.sum() > 10:
                rho.append(spearmanr(pos[sel], ang[sel]).correlation)

            # B. do we add roughness or remove it?
            lc = np.linalg.norm(L @ comp, axis=1)
            lg = np.linalg.norm(L @ gt, axis=1)
            rough_ratio.append(lc[mask].mean() / max(lg[mask].mean(), 1e-12))

            # C. how much of the error field is high-frequency?
            e = comp - gt
            le = np.linalg.norm(L @ e, axis=1)
            en = np.linalg.norm(e, axis=1)
            err_rough.append(le[mask].mean() / max(en[mask].mean(), 1e-12))
            # same ratio for the GT surface = the "natural" roughness scale
            gt_rough.append(lg[mask].mean() /
                            max(np.linalg.norm(gt - gt.mean(0), axis=1)[mask].mean(), 1e-12))

        print(f"\n=== {arm} ===")
        print(f"A. Spearman rho(position err, normal err) over hole : "
              f"{np.mean(rho):.3f} +/- {np.std(rho):.3f}")
        print(f"B. roughness ratio |L(ours)| / |L(gt)| over hole    : "
              f"{np.mean(rough_ratio):.3f} +/- {np.std(rough_ratio):.3f}")
        print(f"C. error-field roughness |L(e)|/|e|                 : "
              f"{np.mean(err_rough):.3f} +/- {np.std(err_rough):.3f}")
        print(f"   (GT surface's own |L(v)|/|v-c| for scale        : "
              f"{np.mean(gt_rough):.4f})")


if __name__ == '__main__':
    main()
