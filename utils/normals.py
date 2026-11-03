"""
Surface-orientation metrics for completion arms (Olivier, 2026-07-30 thread).

Positional L2 is a zeroth-order metric: it says where each vertex is, and says
nothing about how the surface is TILTED there. Two completions can sit within a
few hundredths of a millimetre of each other in position and still differ
visibly, because shading is computed from normals and normals are built from
DIFFERENCES between neighbouring vertices. Small position errors that alternate
direction between neighbours are nearly free under L2 and wreck the normals.

This matters here for two specific reasons:

  1. The seam. DDIM sampling clamps known vertices to their exact values, so
     there is no position gap at the hole boundary -- by construction. What can
     remain is the completed surface ARRIVING at the seam at the wrong angle, a
     crease. No positional metric can see it, because the positions are already
     exact. `boundary_band()` isolates exactly that region.

  2. The regressor comparison. An MSE regressor predicts the conditional MEAN of
     all plausible completions, which is over-smoothed -- that is precisely why
     it wins on L2 (1.04 vs 1.36 mm). Over-smoothing should COST it on normals.
     So this metric is where diffusion may already be ahead without us knowing.

Everything here assumes dense correspondence (fixed topology, vertex i <-> i)
and a shared face list, which holds for COMA/FLAME and DFAUST registrations.
"""

import numpy as np

EPS = 1e-12


def face_normals(verts, faces, normalize=False):
    """
    Per-face normals from the triangle's two edge vectors.

    Left UNNORMALIZED by default: ||(b-a) x (c-a)|| is twice the triangle area,
    so summing raw cross products onto vertices gives area weighting for free.
    Large triangles should dominate a vertex's normal; normalizing first would
    give a sliver the same vote as the face beside it.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if normalize:
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + EPS)
    return n


def vertex_normals(verts, faces):
    """
    Area-weighted unit vertex normals, (N,3).

    Degenerate vertices (zero-area incident faces, or an isolated vertex with no
    faces at all) would divide by ~0; they are returned as exact zeros so the
    caller can detect and exclude them rather than silently comparing garbage
    directions. `normal_angular_error` treats them as invalid.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = face_normals(v, f)                       # (F,3), area-weighted

    vn = np.zeros_like(v)
    for k in range(3):                            # scatter-add onto the 3 corners
        np.add.at(vn, f[:, k], fn)

    mag = np.linalg.norm(vn, axis=1, keepdims=True)
    good = mag[:, 0] > EPS
    vn[good] = vn[good] / mag[good]
    vn[~good] = 0.0
    return vn


def normal_angular_error(verts_pred, verts_gt, faces):
    """
    Per-vertex angle between predicted and GT vertex normals, in DEGREES.

    Returns (err_deg, valid): err_deg is NaN where either normal is degenerate,
    and `valid` flags the usable vertices. Degrees rather than raw cosine
    because "the surface is tilted 3.4 degrees off" is a quantity a reader can
    picture; the cosine form (normal consistency) is reported alongside it in
    `normal_stats`, since that is what the literature tabulates.
    """
    n_pred = vertex_normals(verts_pred, faces)
    n_gt = vertex_normals(verts_gt, faces)

    valid = (np.linalg.norm(n_pred, axis=1) > EPS) & (np.linalg.norm(n_gt, axis=1) > EPS)
    dot = np.clip((n_pred * n_gt).sum(1), -1.0, 1.0)

    err = np.full(len(dot), np.nan)
    err[valid] = np.degrees(np.arccos(dot[valid]))
    return err, valid


def boundary_band(adjacency, missing_mask, rings=2):
    """
    Vertices within `rings` hops of the hole's seam, on BOTH sides.

    The crease lives exactly at the known/missing interface, so a band that only
    covers missing vertices would measure the thing next to the defect rather
    than the defect. Seeded from the interface itself and dilated outward, the
    band therefore straddles it.

    `adjacency` is a sparse (N,N) 0/1 matrix -- pass `BoundaryFieldSolver._A`,
    which is already built and cached from the template.
    """
    m = np.asarray(missing_mask, dtype=bool)
    if not m.any() or m.all():
        return np.zeros(len(m), dtype=bool)

    # Seed: vertices with a neighbour on the other side of the mask (both sides).
    touch_missing = (adjacency @ m.astype(np.float64)) > 0
    touch_known = (adjacency @ (~m).astype(np.float64)) > 0
    band = (touch_missing & ~m) | (touch_known & m)

    for _ in range(max(0, rings - 1)):             # dilate by one hop at a time
        band = band | ((adjacency @ band.astype(np.float64)) > 0)
    return band


def normal_stats(err_deg, valid, sel, prefix):
    """mean/median/p95 angular error (deg) + mean normal consistency over `sel`."""
    use = sel & valid
    if not use.any():
        return {f'{prefix}_deg_mean': 0.0, f'{prefix}_deg_median': 0.0,
                f'{prefix}_deg_p95': 0.0, f'{prefix}_nc': 1.0, f'{prefix}_n': 0}
    e = err_deg[use]
    return {f'{prefix}_deg_mean': float(e.mean()),
            f'{prefix}_deg_median': float(np.median(e)),
            f'{prefix}_deg_p95': float(np.percentile(e, 95)),
            # Normal consistency = mean cos(angle), the form papers tabulate.
            f'{prefix}_nc': float(np.cos(np.radians(e)).mean()),
            f'{prefix}_n': int(use.sum())}


# ---------------------------------------------------------------------------
def _self_test():
    """Check the metric on the real COMA template with synthetic perturbations."""
    import os
    import trimesh
    from utils.boundary_field import BoundaryFieldSolver
    from utils.masking import generate_random_mask, _build_adjacency

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = trimesh.load(os.path.join(here, 'data', 'template.ply'), process=False)
    V, F = np.asarray(tpl.vertices), np.asarray(tpl.faces)
    solver = BoundaryFieldSolver(F, V)

    # Identical meshes -> zero error. Tolerance is 1e-4 deg, not 0: arccos is
    # ill-conditioned near dot=1 (angle ~ sqrt(2*eps)), so double-precision
    # round-off alone lands around 1e-6 deg. That is numerically exact here.
    err, valid = normal_angular_error(V, V, F)
    assert np.nanmax(err) < 1e-4, f"identical meshes gave {np.nanmax(err)} deg"
    print(f"identical: max {np.nanmax(err):.2e} deg, {valid.sum()}/{len(V)} valid")

    # Agreement with trimesh's own vertex normals (sanity on orientation).
    ours = vertex_normals(V, F)
    theirs = np.asarray(tpl.vertex_normals)
    good = np.linalg.norm(ours, axis=1) > EPS
    cos = (ours[good] * theirs[good]).sum(1)
    print(f"vs trimesh vertex_normals: mean cos {cos.mean():.6f} (expect ~1)")
    assert cos.mean() > 0.99

    # The key property: a RIGID shift has huge L2 error and ZERO normal error,
    # while alternating per-vertex jitter is the reverse. This is the whole
    # argument for adding the metric, so it is worth asserting.
    shift = V + np.array([0.01, 0.0, 0.0])          # 10 mm bodily translation
    e_shift, _ = normal_angular_error(shift, V, F)
    rng = np.random.RandomState(0)
    sign = rng.choice([-1.0, 1.0], size=len(V))[:, None]
    jitter = V + sign * vertex_normals(V, F) * 1e-4  # 0.1 mm, alternating
    e_jit, _ = normal_angular_error(jitter, V, F)
    print(f"rigid 10 mm shift : L2 10.000 mm, normals {np.nanmean(e_shift):.4f} deg")
    print(f"alternating 0.1 mm: L2  0.100 mm, normals {np.nanmean(e_jit):.4f} deg")
    assert np.nanmean(e_shift) < 1e-6
    # ~2 deg, not more: a vertex normal averages ~6 incident faces, and against
    # COMA's ~4 mm edges a 0.1 mm ripple subtends only a couple of degrees. The
    # point is the CONTRAST -- 100x less L2 error, unboundedly more normal error.
    assert np.nanmean(e_jit) > 1.0
    assert np.nanmean(e_jit) > 100 * np.nanmean(e_shift)

    # Boundary band on a real mask. NOTE seed 0 is deliberately avoided: it
    # produces a 546-vertex mask that is exactly one whole eyeball, and the
    # eyeballs are disconnected components of this template, so that mask has NO
    # seam at all and an empty band is the correct answer (asserted below).
    adj = _build_adjacency(np.asarray(F, dtype=np.int64), len(V))
    mask, strat, ratio = generate_random_mask(
        V, F, mask_ratio_range=(0.05, 0.40),
        strategies=('bfs_patch',), strategy_weights=(1.0,),
        rng=np.random.RandomState(1), adj=adj)
    print(f"mask [{strat} r={ratio:.2f}] missing={int(mask.sum())} "
          f"ring={len(solver.boundary_ring(mask))}")
    for r in (1, 2, 3):
        band = boundary_band(solver._A, mask, rings=r)
        both = (band & mask).sum(), (band & ~mask).sum()
        print(f"band rings={r}: {band.sum():4d} verts "
              f"({both[0]} missing + {both[1]} known)")
        assert band.sum() > 0 and both[0] > 0 and both[1] > 0, "band must straddle the seam"

    # Empty / full masks must not produce a band.
    assert not boundary_band(solver._A, np.zeros(len(V), bool)).any()
    assert not boundary_band(solver._A, np.ones(len(V), bool)).any()
    print("empty/full mask -> empty band")

    # A fully-missing disconnected component (whole eyeball) has no seam, so the
    # band must be empty rather than wrong. This fires on real eval data.
    eye, _, _ = generate_random_mask(
        V, F, mask_ratio_range=(0.05, 0.40),
        strategies=('bfs_patch',), strategy_weights=(1.0,),
        rng=np.random.RandomState(0), adj=adj)
    assert int(eye.sum()) == 546 and len(solver.boundary_ring(eye)) == 0
    assert not boundary_band(solver._A, eye).any()
    print("whole-eyeball mask (no seam) -> empty band")
    print("\nself-test OK")


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _self_test()
