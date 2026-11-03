"""
Post-processing for mesh completion results.

Applies boundary-aware Laplacian smoothing to eliminate the seam artifact
between known (original) and predicted (completed) vertices.

The idea: only smooth vertices near the boundary between known and missing
regions. This removes the discontinuity while preserving the original
known geometry and the predicted interior geometry.
"""

import numpy as np
from collections import defaultdict


def build_adjacency_dict(faces, num_vertices):
    """Build vertex adjacency dictionary from faces."""
    adj = defaultdict(set)
    for f in faces:
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                adj[f[i]].add(f[j])
                adj[f[j]].add(f[i])
    return adj


def find_boundary_band(missing_mask, faces, num_rings=3):
    """
    Find vertices in the boundary band between known and missing regions.

    Returns a float weight array (N,) where:
      0.0 = far from boundary (don't smooth)
      1.0 = right at the boundary (smooth most)
    Weights decay linearly with ring distance from the boundary.

    Args:
        missing_mask: (N,) bool, True = missing vertex
        faces: (F, 3) int array
        num_rings: how many rings around the boundary to smooth

    Returns:
        smooth_weights: (N,) float array in [0, 1]
    """
    N = len(missing_mask)
    adj = build_adjacency_dict(faces, N)

    # Find boundary vertices: missing vertices that have at least one known neighbor,
    # OR known vertices that have at least one missing neighbor
    boundary_set = set()
    for v in range(N):
        if missing_mask[v]:
            for nb in adj[v]:
                if not missing_mask[nb]:
                    boundary_set.add(v)
                    boundary_set.add(nb)
                    break
        else:
            for nb in adj[v]:
                if missing_mask[nb]:
                    boundary_set.add(v)
                    boundary_set.add(nb)
                    break

    # BFS expansion from boundary to find rings
    ring_distance = np.full(N, -1, dtype=int)
    for v in boundary_set:
        ring_distance[v] = 0

    current_ring = list(boundary_set)
    for ring in range(1, num_rings + 1):
        next_ring = []
        for v in current_ring:
            for nb in adj[v]:
                if ring_distance[nb] == -1:
                    ring_distance[nb] = ring
                    next_ring.append(nb)
        current_ring = next_ring

    # Build smooth weights: linearly decay from 1.0 at boundary to 0.0 at num_rings
    smooth_weights = np.zeros(N, dtype=np.float32)
    for v in range(N):
        if ring_distance[v] >= 0:
            smooth_weights[v] = 1.0 - (ring_distance[v] / (num_rings + 1))

    return smooth_weights


def laplacian_smooth(vertices, faces, smooth_weights, iterations=5, lambda_factor=0.5):
    """
    Weighted Laplacian smoothing.

    Only moves vertices where smooth_weights > 0, and by an amount
    proportional to the weight. This preserves known geometry far from
    the boundary while smoothing the seam.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        smooth_weights: (N,) float in [0, 1], how much each vertex should be smoothed
        iterations: number of smoothing passes
        lambda_factor: step size per iteration (0.3–0.7 typical)

    Returns:
        smoothed_vertices: (N, 3) smoothed positions
    """
    adj = build_adjacency_dict(faces, len(vertices))
    verts = vertices.copy()

    for _ in range(iterations):
        new_verts = verts.copy()
        for v in range(len(verts)):
            w = smooth_weights[v]
            if w < 1e-6:
                continue
            neighbors = list(adj[v])
            if len(neighbors) == 0:
                continue
            # Laplacian: average of neighbors minus current position
            neighbor_avg = np.mean(verts[neighbors], axis=0)
            displacement = neighbor_avg - verts[v]
            new_verts[v] = verts[v] + w * lambda_factor * displacement
        verts = new_verts

    return verts


def taubin_smooth(vertices, faces, smooth_weights, iterations=5,
                  lambda_factor=0.5, mu_factor=-0.53):
    """
    Taubin smoothing (lambda-mu): alternating positive and negative
    Laplacian steps to smooth without shrinkage.

    Args:
        vertices: (N, 3)
        faces: (F, 3)
        smooth_weights: (N,) float in [0, 1]
        iterations: number of Taubin iterations (each = 2 Laplacian steps)
        lambda_factor: positive step
        mu_factor: negative step (should be slightly more negative than -lambda)

    Returns:
        smoothed_vertices: (N, 3)
    """
    adj = build_adjacency_dict(faces, len(vertices))
    verts = vertices.copy()

    for _ in range(iterations):
        # Lambda step (shrink)
        verts = _laplacian_step(verts, adj, smooth_weights, lambda_factor)
        # Mu step (inflate back)
        verts = _laplacian_step(verts, adj, smooth_weights, mu_factor)

    return verts


def _laplacian_step(verts, adj, smooth_weights, factor):
    """Single weighted Laplacian step."""
    new_verts = verts.copy()
    for v in range(len(verts)):
        w = smooth_weights[v]
        if w < 1e-6:
            continue
        neighbors = list(adj[v])
        if len(neighbors) == 0:
            continue
        neighbor_avg = np.mean(verts[neighbors], axis=0)
        displacement = neighbor_avg - verts[v]
        new_verts[v] = verts[v] + w * factor * displacement
    return new_verts


def boundary_stitch(vertices, faces, missing_mask, num_rings=3,
                    base_stitch_weight=0.7):
    """
    Boundary stitching inspired by classical mesh completion pipelines.

    Instead of uniform Laplacian smoothing, this method explicitly pulls
    missing boundary vertices toward their known neighbors, ensuring
    geometric (C0) continuity at the known/predicted interface.

    The algorithm processes ring-by-ring outward from the boundary:
      Ring 0: missing vertices directly adjacent to known vertices
              → blended toward average of known neighbors
      Ring 1: missing vertices adjacent to ring-0 vertices
              → blended toward average of known + already-stitched ring-0 neighbors
      Ring k: uses anchors from rings 0..k-1

    The stitch weight decays linearly with ring distance:
      w(ring) = base_stitch_weight * (1 - ring / num_rings)

    For each vertex:
      new_pos = (1 - w) * predicted_pos + w * avg(anchor_neighbors)

    Args:
        vertices: (N, 3) completed vertex positions (after model blending)
        faces: (F, 3) face connectivity
        missing_mask: (N,) bool, True = missing (predicted by model)
        num_rings: how many rings into the missing region to stitch
        base_stitch_weight: blend weight at ring 0 (0.0 = keep prediction, 1.0 = full anchor)

    Returns:
        stitched_vertices: (N, 3) stitched positions
        num_stitched: number of vertices that were modified
    """
    N = len(vertices)
    adj = build_adjacency_dict(faces, N)
    verts = vertices.copy()

    # --- Step 1: Find ring-0 missing vertices (adjacent to known) ---
    ring_assignment = np.full(N, -1, dtype=int)
    ring_0 = []
    for v in range(N):
        if missing_mask[v]:
            for nb in adj[v]:
                if not missing_mask[nb]:
                    ring_assignment[v] = 0
                    ring_0.append(v)
                    break

    # --- Step 2: BFS expand into missing region to find rings 1..num_rings-1 ---
    current = ring_0
    for ring in range(1, num_rings):
        next_ring = []
        for v in current:
            for nb in adj[v]:
                if missing_mask[nb] and ring_assignment[nb] == -1:
                    ring_assignment[nb] = ring
                    next_ring.append(nb)
        current = next_ring

    # --- Step 3: Stitch ring-by-ring (order matters!) ---
    # Process inner rings first so outer rings can use them as anchors.
    for ring in range(num_rings):
        ring_verts = [v for v in range(N) if ring_assignment[v] == ring]

        # Stitch weight decays with ring distance
        decay = 1.0 - (ring / num_rings)
        w = base_stitch_weight * decay

        for v in ring_verts:
            # Gather anchor neighbors:
            #   - Known vertices (original, never changed)
            #   - Already-stitched vertices from previous rings
            anchor_nbs = []
            for nb in adj[v]:
                if not missing_mask[nb]:
                    # Known vertex — strongest anchor
                    anchor_nbs.append(nb)
                elif ring_assignment[nb] >= 0 and ring_assignment[nb] < ring:
                    # Previously stitched missing vertex
                    anchor_nbs.append(nb)

            if not anchor_nbs:
                continue

            anchor_avg = np.mean(verts[anchor_nbs], axis=0)
            verts[v] = (1.0 - w) * verts[v] + w * anchor_avg

    num_stitched = int((ring_assignment >= 0).sum())
    return verts, num_stitched


def postprocess_completion(vertices, faces, missing_mask,
                           method='taubin', num_rings=3, iterations=5,
                           lambda_factor=0.5, mu_factor=-0.53,
                           stitch_weight=0.7):
    """
    Full post-processing pipeline for a completed mesh.

    Args:
        vertices: (N, 3) completed vertex positions (after blending)
        faces: (F, 3) face connectivity
        missing_mask: (N,) bool, True = was missing (predicted by model)
        method: 'laplacian', 'taubin', 'stitch', or 'stitch+taubin'
        num_rings: how many rings around the boundary to process
        iterations: smoothing iterations (for laplacian/taubin steps)
        lambda_factor: smoothing step size
        mu_factor: Taubin inflation step (only used if method includes 'taubin')
        stitch_weight: base stitching weight (only used if method includes 'stitch')

    Returns:
        processed_vertices: (N, 3) post-processed positions
        num_affected: number of vertices that were modified
    """
    if method == 'stitch':
        # Pure boundary stitching — no smoothing
        result, num_affected = boundary_stitch(
            vertices, faces, missing_mask,
            num_rings=num_rings, base_stitch_weight=stitch_weight
        )
        return result, num_affected

    elif method == 'stitch+taubin':
        # Boundary stitching first, then light Taubin to polish
        stitched, num_stitched = boundary_stitch(
            vertices, faces, missing_mask,
            num_rings=num_rings, base_stitch_weight=stitch_weight
        )
        smooth_weights = find_boundary_band(missing_mask, faces, num_rings=num_rings)
        result = taubin_smooth(stitched, faces, smooth_weights,
                               iterations=iterations,
                               lambda_factor=lambda_factor,
                               mu_factor=mu_factor)
        num_affected = int((smooth_weights > 0).sum())
        return result, num_affected

    else:
        # Original behavior: pure smoothing
        smooth_weights = find_boundary_band(missing_mask, faces, num_rings=num_rings)
        num_affected = int((smooth_weights > 0).sum())

        if method == 'taubin':
            result = taubin_smooth(vertices, faces, smooth_weights,
                                   iterations=iterations,
                                   lambda_factor=lambda_factor,
                                   mu_factor=mu_factor)
        else:
            result = laplacian_smooth(vertices, faces, smooth_weights,
                                      iterations=iterations,
                                      lambda_factor=lambda_factor)

        return result, num_affected
