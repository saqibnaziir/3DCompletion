"""
Masking utilities for generating synthetic partial meshes from complete meshes.

Three strategies:
  1. BFS patch  – pick a random seed vertex, BFS-expand to mask a contiguous region
  2. Plane cut  – random plane through the mesh, mask all vertices on one side
  3. Random scatter – randomly drop individual vertices

All functions return a boolean mask of shape (N,) where True = MISSING.
"""

import numpy as np
from collections import deque


def _build_adjacency(faces, num_vertices):
    """Build vertex adjacency list from face array."""
    adj = [[] for _ in range(num_vertices)]
    for f in faces:
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                adj[f[i]].append(f[j])
                adj[f[j]].append(f[i])
    return adj


def mask_bfs_patch(vertices, faces, mask_ratio=0.25, rng=None, adj=None):
    """
    Mask a contiguous patch via BFS expansion from a random seed vertex.

    Args:
        vertices: (N, 3) numpy array
        faces: (F, 3) numpy array
        mask_ratio: fraction of vertices to mask (0, 1)
        rng: numpy RandomState for reproducibility
        adj: optional precomputed adjacency list (avoids rebuilding it every call,
             which is a major CPU cost for on-the-fly masking on fixed topology)

    Returns:
        mask: (N,) bool array, True = missing
    """
    if rng is None:
        rng = np.random.RandomState()

    N = len(vertices)
    target = max(1, int(N * mask_ratio))
    if adj is None:
        adj = _build_adjacency(faces, N)

    seed = rng.randint(0, N)
    visited = set()
    queue = deque([seed])
    visited.add(seed)

    while len(visited) < target and queue:
        v = queue.popleft()
        neighbors = list(adj[v])
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target:
                    break

    mask = np.zeros(N, dtype=bool)
    mask[list(visited)] = True
    return mask


def mask_plane_cut(vertices, faces, mask_ratio=0.25, rng=None, adj=None):
    """
    Mask vertices on one side of a random plane through the mesh centroid.
    The plane orientation is random; if the resulting mask ratio is far from
    the target, we shift the plane to get closer.

    Args:
        vertices: (N, 3) numpy array
        faces: (F, 3) numpy array  (unused, kept for consistent API)
        mask_ratio: approximate fraction to mask
        rng: numpy RandomState

    Returns:
        mask: (N,) bool array, True = missing
    """
    if rng is None:
        rng = np.random.RandomState()

    N = len(vertices)
    centroid = vertices.mean(axis=0)

    # Random plane normal
    normal = rng.randn(3)
    normal /= np.linalg.norm(normal) + 1e-8

    # Project vertices onto the normal direction
    projections = (vertices - centroid) @ normal  # (N,)

    # Find the threshold that gives approximately mask_ratio fraction
    sorted_proj = np.sort(projections)
    # We want the top mask_ratio fraction
    idx = max(0, int(N * (1.0 - mask_ratio)))
    threshold = sorted_proj[min(idx, N - 1)]

    mask = projections >= threshold
    return mask


def mask_random_scatter(vertices, faces, mask_ratio=0.15, rng=None, adj=None):
    """
    Randomly mask individual vertices (scattered dropout).

    Args:
        vertices: (N, 3) numpy array
        faces: (F, 3) numpy array  (unused, kept for consistent API)
        mask_ratio: fraction to mask
        rng: numpy RandomState

    Returns:
        mask: (N,) bool array, True = missing
    """
    if rng is None:
        rng = np.random.RandomState()

    N = len(vertices)
    mask = rng.rand(N) < mask_ratio
    # Ensure at least one vertex is masked and at least one is kept
    if mask.sum() == 0:
        mask[rng.randint(0, N)] = True
    if mask.sum() == N:
        mask[rng.randint(0, N)] = False
    return mask


# ---- Main entry point used by the dataset ----

MASK_STRATEGIES = {
    'bfs_patch': mask_bfs_patch,
    'plane_cut': mask_plane_cut,
    'random_scatter': mask_random_scatter,
}


def generate_random_mask(vertices, faces, mask_ratio_range=(0.05, 0.15),
                         strategies=('bfs_patch', 'plane_cut', 'random_scatter'),
                         strategy_weights=(0.5, 0.3, 0.2), rng=None, adj=None):
    """
    Generate a random mask using a randomly chosen strategy and ratio.

    Args:
        vertices: (N, 3) numpy array
        faces: (F, 3) numpy array
        mask_ratio_range: (min, max) for the mask ratio
        strategies: tuple of strategy names to choose from
        strategy_weights: probability weights for each strategy
        rng: numpy RandomState

    Returns:
        mask: (N,) bool array, True = missing
        strategy_name: str, which strategy was used
        mask_ratio: float, actual ratio of masked vertices
    """
    if rng is None:
        rng = np.random.RandomState()

    # Pick a random ratio
    lo, hi = mask_ratio_range
    mask_ratio = rng.uniform(lo, hi)

    # Pick a random strategy
    weights = np.array(strategy_weights, dtype=float)
    weights /= weights.sum()
    strategy_idx = rng.choice(len(strategies), p=weights)
    strategy_name = strategies[strategy_idx]

    mask_fn = MASK_STRATEGIES[strategy_name]
    mask = mask_fn(vertices, faces, mask_ratio=mask_ratio, rng=rng, adj=adj)

    actual_ratio = mask.sum() / len(mask)
    return mask, strategy_name, actual_ratio
