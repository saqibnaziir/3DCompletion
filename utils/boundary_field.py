"""
Geodesic depth-weighting of the completion loss (Olivier, 2026-07-25 thread).

Instead of a flat `lambda_missing` over the whole hole, weight each missing
vertex by how deep inside the hole it sits:

    d(v)                geodesic distance from missing vertex v to the nearest
                        KNOWN boundary vertex
    d_max               max of d over the missing region (per mask)
    w(v) = 1 + (w_max - 1) * d(v) / d_max      for missing vertices
    w(v) = 1                                    for known vertices

so w -> w_max (default 3) at the deepest, hardest points and w -> 1 right next
to the seam. This matches what the C2 benchmark already showed: reconstruction
error grows with hole DEPTH, and boundary vertices are the easy ones. The flat
weight currently spends the same effort on both.

Backends for d(v):
  'dijkstra'  (default) -- exact shortest path on the mesh edge graph, weighted
                by template edge lengths. Pure scipy, no extra dependency.
  'hops'      -- unweighted BFS hop count x mean edge length. Slightly faster,
                coarser.
  'euclidean' -- straight-line distance to the boundary (KDTree). Ignores the
                surface; the thread calls this "less interesting".
  'heat'      -- potpourri3d heat method (true smooth geodesic). Used only if
                the library imports. It is NOT installed in the `3dmesh` env
                and pip there fails on the Windows truststore SSL bug, so
                'dijkstra' is the practical default.

TOPOLOGY GOTCHA (COMA/FLAME): the template is NOT connected -- it is 3 pieces,
the face (3931 verts) plus two separate eyeballs (546 each). A boundary ring on
the face can never reach an eyeball, so those vertices come back as +inf. Left
alone that would produce NaN weights and poison the gradient, so unreachable
missing vertices are treated as maximally deep (w = w_max): their whole
component is missing, meaning there is no local context at all, which is
exactly the "hardest" case the ramp is meant to emphasise.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

try:  # optional, see module docstring
    import potpourri3d as pp3d
    HAS_POTPOURRI = True
except ImportError:
    HAS_POTPOURRI = False

BACKENDS = ('dijkstra', 'hops', 'euclidean', 'heat')


class BoundaryFieldSolver:
    """
    Precomputes the mesh edge graph once, then answers per-mask distance
    queries cheaply (~1.5 ms per mask on COMA's 5023 vertices, so ~4 s/epoch
    spread across the dataloader workers -- hidden behind the GPU step).

    The graph is built from the FIXED template connectivity and template edge
    lengths. The topology never changes and per-sample positions differ only by
    a small expression offset, so reusing the template metric is what makes
    on-the-fly masking in __getitem__ affordable.
    """

    def __init__(self, faces, verts, backend='dijkstra'):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if backend == 'heat' and not HAS_POTPOURRI:
            raise ImportError(
                "backend='heat' needs potpourri3d (pip install potpourri3d). "
                "Use backend='dijkstra' for an exact edge-graph geodesic with "
                "no extra dependency.")

        self.backend = backend
        self.faces = np.asarray(faces, dtype=np.int64)
        self.verts = np.asarray(verts, dtype=np.float64)
        self.n = int(self.verts.shape[0])

        # Undirected edge list (both directions) from the triangle list.
        e = np.vstack([self.faces[:, [0, 1]],
                       self.faces[:, [1, 2]],
                       self.faces[:, [2, 0]]])
        e = np.vstack([e, e[:, ::-1]])

        length = np.linalg.norm(self.verts[e[:, 0]] - self.verts[e[:, 1]], axis=1)
        self.mean_edge = float(length[length > 0].mean())
        self.bbox_diag = float(np.linalg.norm(self.verts.max(0) - self.verts.min(0)))

        # Weighted graph (metric) and unweighted graph (adjacency / hops).
        self._G = sp.csr_matrix((length, (e[:, 0], e[:, 1])), shape=(self.n, self.n))
        self._A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                                shape=(self.n, self.n))

        self._heat_solver = None  # built lazily, it is the expensive one

    # -- boundary ring ----------------------------------------------------
    def boundary_ring(self, missing_mask):
        """
        Known vertices that touch a missing vertex -- the seam, and the zero
        level set of the distance field.

        Taking the ring on the KNOWN side is what Olivier specified: d(v) is
        measured from missing vertices to "the known boundary". Missing
        vertices adjacent to the ring therefore sit one edge inside at d>0, so
        w is just above 1 there rather than exactly 1.
        """
        missing_mask = np.asarray(missing_mask, dtype=bool)
        hit = (sp.csr_matrix(missing_mask.astype(np.float64).reshape(1, -1))
               @ self._A).toarray().ravel() > 0
        return np.where(hit & ~missing_mask)[0]

    # -- distance ---------------------------------------------------------
    def depth(self, missing_mask):
        """
        Geodesic depth of each missing vertex below the known boundary.

        Travel is restricted to the hole (plus its ring), so the path cannot
        leak out around the known surface and re-enter elsewhere.

        Returns (d, ring): d is (n,), finite on missing vertices reachable from
        the ring, +inf on known vertices and on hole components that have no
        known neighbour at all.
        """
        missing_mask = np.asarray(missing_mask, dtype=bool)
        ring = self.boundary_ring(missing_mask)

        d = np.full(self.n, np.inf)
        if len(ring) == 0 or not missing_mask.any():
            return d, ring

        on_ring = np.zeros(self.n, dtype=bool)
        on_ring[ring] = True
        allowed = missing_mask | on_ring

        keep = np.where(allowed)[0]
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        src = remap[ring]

        if self.backend == 'euclidean':
            from scipy.spatial import cKDTree
            sub_d = cKDTree(self.verts[ring]).query(self.verts[keep])[0]
        elif self.backend == 'heat':
            if self._heat_solver is None:
                self._heat_solver = pp3d.MeshHeatMethodDistanceSolver(
                    self.verts, self.faces)
            # The heat method runs on the whole mesh; it cannot be restricted
            # to a subset, so `allowed` is applied afterwards by slicing.
            sub_d = self._heat_solver.compute_distance_multisource(ring)[keep]
        else:
            sub = (self._A if self.backend == 'hops' else self._G)[keep][:, keep]
            sub_d = dijkstra(sub, indices=src, min_only=True, directed=False,
                             unweighted=(self.backend == 'hops'))
            if self.backend == 'hops':
                sub_d = sub_d * self.mean_edge

        d[keep] = sub_d
        d[~missing_mask] = np.inf      # depth is only defined inside the hole
        return d, ring

    # -- Olivier's depth-ramped loss weight --------------------------------
    def depth_weight(self, missing_mask, w_max=3.0, w_known=1.0,
                     normalize=False, d=None):
        """
        w(v) = w_known + (w_max - w_known) * d(v)/d_max   on missing vertices
        w(v) = w_known                                    on known vertices

        normalize: rescale the missing-region weights so their MEAN equals
            w_max. Off by default (that is the formula verbatim). It is worth
            having because the ramp averages ~w_max/2 over the hole, so against
            today's flat lambda_missing=3.0 an un-normalized ramp changes both
            the SHAPE of the weighting and the overall missing-vs-known
            emphasis at once. Turning it on isolates the shape effect, which is
            the thing actually under test.

        Always returns finite values in [w_known, w_max].
        """
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if d is None:
            d, _ = self.depth(missing_mask)

        w = np.full(self.n, float(w_known), dtype=np.float64)
        if not missing_mask.any():
            return w

        dm = d[missing_mask]
        finite = np.isfinite(dm)
        d_max = dm[finite].max() if finite.any() else 0.0

        if d_max <= 0:
            # Hole is one ring-deep, or fully disconnected from any known
            # vertex: there is no depth to ramp over, so use flat max emphasis.
            q = np.ones_like(dm)
        else:
            q = np.where(finite, dm / d_max, 1.0)   # unreachable => deepest

        w_missing = w_known + (w_max - w_known) * q
        if normalize:
            m = w_missing.mean()
            if m > 0:
                w_missing = w_missing * (w_max / m)

        w[missing_mask] = w_missing
        return w


# ---------------------------------------------------------------------------
def _self_test():
    """Sanity-check the weight field on the real COMA template."""
    import os
    import time
    import trimesh
    from scipy.sparse.csgraph import connected_components
    from utils.masking import generate_random_mask, _build_adjacency

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = trimesh.load(os.path.join(here, 'data', 'template.ply'), process=False)
    V, F = np.asarray(tpl.vertices), np.asarray(tpl.faces)
    solver = BoundaryFieldSolver(F, V)
    print(f"template {solver.n} verts | mean_edge {solver.mean_edge*1000:.2f} mm "
          f"| bbox diag {solver.bbox_diag*1000:.0f} mm")

    adj = _build_adjacency(np.asarray(F, dtype=np.int64), solver.n)
    rng = np.random.RandomState(0)

    t0 = time.perf_counter()
    for i in range(20):
        mask, strat, ratio = generate_random_mask(
            V, F, mask_ratio_range=(0.05, 0.40),
            strategies=('bfs_patch', 'plane_cut', 'random_scatter'),
            strategy_weights=(0.5, 0.3, 0.2), rng=rng, adj=adj)
        d, ring = solver.depth(mask)
        w = solver.depth_weight(mask, w_max=3.0, d=d)

        assert np.isfinite(w).all(), "weights must be finite"
        assert (w[~mask] == 1.0).all(), "known weights must stay exactly 1"
        assert w.min() >= 1.0 - 1e-9, f"w below w_known: {w.min()}"
        assert w.max() <= 3.0 + 1e-9, f"w above w_max: {w.max()}"
        if i < 6:
            fin = np.isfinite(d)
            print(f"  [{strat:14s} r={ratio:.2f}] missing={int(mask.sum()):5d} "
                  f"ring={len(ring):4d} depth_max={d[fin].max()*1000:6.1f} mm "
                  f"w_missing mean={w[mask].mean():.2f} max={w[mask].max():.2f}")
    dt = (time.perf_counter() - t0) / 20
    print(f"\ndepth+weight: {dt*1000:.2f} ms/sample")

    # Normalization option: mean over the hole should land on w_max.
    wn = solver.depth_weight(mask, w_max=3.0, normalize=True)
    print(f"normalize=True -> w_missing mean={wn[mask].mean():.3f} (expect 3.000), "
          f"max={wn[mask].max():.2f}")
    assert abs(wn[mask].mean() - 3.0) < 1e-6

    # A fully-missing eyeball has no known neighbour -> must not produce NaN.
    _, lab = connected_components(solver._A, directed=False)
    eye = lab == 1
    w_eye = solver.depth_weight(eye, w_max=3.0)
    assert np.isfinite(w_eye).all()
    print(f"disconnected-eyeball hole: ring={len(solver.boundary_ring(eye))} "
          f"w[eye]={w_eye[eye].min():.2f}..{w_eye[eye].max():.2f} "
          f"(expect flat 3.00 = maximally deep, no context)")

    # Empty mask must be a no-op.
    w0 = solver.depth_weight(np.zeros(solver.n, dtype=bool), w_max=3.0)
    assert (w0 == 1.0).all()
    print("empty mask -> all weights 1.00")
    print("\nself-test OK")


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _self_test()
