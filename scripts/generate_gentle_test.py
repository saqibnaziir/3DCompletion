"""
Generate a GENTLE completion test set: small, localized contiguous holes.

Motivation
----------
The default test set uses plane_cut / large-ratio bfs_patch masks, which amputate
whole limbs (a full leg / arm / head). For those, many completions are equally
valid and per-vertex L2 to a single GT is unfair (see the qualitative figures:
the filled limb is plausible but in a different pose -> large error).

This script builds an easier, more realistic scan-hole test set:
  * bfs_patch ONLY (a connected blob grown from a random seed vertex),
  * small ratios (default 3-10% per patch),
  * optionally a few disjoint patches per mesh (arm + leg + torso style),
so each hole is a *local* missing region the model can actually recover, rather
than a whole missing body part.

It is NON-DESTRUCTIVE: GT meshes are read from --src_dir and everything
(GT copies + masks + partials) is written to a fresh --out_dir. The original
test set is left untouched.

Output layout (matches infer_diffusion.py / qualitative_figure.py conventions):
  <out_dir>/test_XXX_gt_complete.ply
  <out_dir>/test_XXX_mask.npy
  <out_dir>/test_XXX_partial_bfs_patch_r<ratio>.ply

Run:
  python scripts/generate_gentle_test.py \
      --src_dir data/DFAUST_ds/completion_test_samples \
      --out_dir data/DFAUST_ds/completion_test_samples_gentle
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import shutil
import numpy as np
from glob import glob
from trimesh import load_mesh, Trimesh

from collections import deque

from utils.funcs import save_ply_explicit
from utils.masking import mask_bfs_patch, _build_adjacency


def vertex_curvature(verts, adj):
    """
    Cheap per-vertex curvature proxy = magnitude of the uniform Laplacian
    (umbrella operator): ||v_i - mean(neighbors)||. Low  -> flat/plain region
    (belly, back, thigh, upper arm); High -> face features, fingers, toes,
    ears, and JOINTS where the surface bends sharply.
    """
    N = len(verts)
    curv = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nb = adj[i]
        if not nb:
            continue
        curv[i] = np.linalg.norm(verts[i] - verts[nb].mean(axis=0))
    return curv


def hop_distance(adj, source):
    """
    BFS edge-hop distance from `source` to every vertex. Because the DFAUST
    topology is fixed, this is pose-invariant and computed ONCE. Extremities
    (hands, fingers, feet, toes, head-top) have the largest hop distance from a
    torso-core seed; the belly/back/chest/thigh/upper-arm are mid-range.
    """
    N = len(adj)
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0
    q = deque([source])
    while q:
        v = q.popleft()
        for j in adj[v]:
            if dist[j] < 0:
                dist[j] = dist[v] + 1
                q.append(j)
    # Any unreached vertex (disconnected) -> treat as far extremity.
    dist[dist < 0] = dist.max()
    return dist


def mask_flat_patch(verts, adj, curv, mask_ratio, rng,
                    seed_pct=30.0, grow_pct=55.0,
                    hop=None, hop_pct=80.0):
    """
    Grow a contiguous patch that stays INSIDE a plain (low-curvature),
    non-extremity region.

      * seed is drawn only from the lowest `seed_pct`% curvature vertices,
      * BFS only expands through vertices below the `grow_pct`% curvature
        threshold, so the hole cannot spill into a joint / face,
      * if `hop` is given, vertices beyond the `hop_pct`% hop-distance
        percentile (hands / feet / fingers / toes / head-top) are excluded
        from BOTH seeding and growth, so holes never land on an extremity.
    """
    N = len(verts)
    target = max(1, int(N * mask_ratio))

    seed_thr = np.percentile(curv, seed_pct)
    grow_thr = np.percentile(curv, grow_pct)
    if hop is not None:
        hop_thr = np.percentile(hop, hop_pct)
        core = hop <= hop_thr            # True = torso/limb-interior (keepable)
    else:
        core = np.ones(N, dtype=bool)

    seed_pool = np.where((curv <= seed_thr) & core)[0]
    if len(seed_pool) == 0:
        seed_pool = np.where(core)[0] if core.any() else np.arange(N)

    seed = seed_pool[rng.randint(0, len(seed_pool))]
    visited = {seed}
    queue = deque([seed])
    while len(visited) < target and queue:
        v = queue.popleft()
        nb = list(adj[v])
        rng.shuffle(nb)
        for j in nb:
            if j not in visited and curv[j] <= grow_thr and core[j]:
                visited.add(j)
                queue.append(j)
                if len(visited) >= target:
                    break

    mask = np.zeros(N, dtype=bool)
    mask[list(visited)] = True
    return mask


def make_gentle_mask(verts, faces, adj, n_patches, ratio_range, rng,
                     region='any', curv=None, hop=None, hop_pct=80.0):
    """Union of `n_patches` small patches grown from distinct seeds.

    region='any'  -> plain bfs patch anywhere (may hit joints/face/hands)
    region='flat' -> patch confined to low-curvature, non-extremity plain areas
    """
    N = len(verts)
    mask = np.zeros(N, dtype=bool)
    for _ in range(n_patches):
        r = rng.uniform(*ratio_range)
        if region == 'flat':
            patch = mask_flat_patch(verts, adj, curv, r, rng,
                                    hop=hop, hop_pct=hop_pct)
        else:
            patch = mask_bfs_patch(verts, faces, mask_ratio=r, rng=rng, adj=adj)
        mask |= patch
    # Safety: never mask everything, always leave context.
    if mask.all():
        mask[rng.randint(0, N)] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, required=True,
                    help="Folder with existing *_gt_complete.ply meshes")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Fresh output folder for the gentle test set")
    ap.add_argument("--ratio_min", type=float, default=0.03,
                    help="Min per-patch mask ratio (fraction of vertices)")
    ap.add_argument("--ratio_max", type=float, default=0.10,
                    help="Max per-patch mask ratio (fraction of vertices)")
    ap.add_argument("--n_patches", type=int, default=1,
                    help="Number of disjoint small patches per mesh")
    ap.add_argument("--region", type=str, default="any", choices=["any", "flat"],
                    help="'flat' confines holes to low-curvature plain areas "
                         "(belly/back/thigh/upper-arm), avoiding face/hands/joints")
    ap.add_argument("--hop_pct", type=float, default=80.0,
                    help="flat mode only: keep vertices within this hop-distance "
                         "percentile from the body core; the rest (hands/feet/"
                         "head-top) are excluded from holes. Lower = stricter.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(args.src_dir, '*_gt_complete.ply')))
    if not gt_files:
        print(f"No *_gt_complete.ply found in {args.src_dir}")
        return
    print(f"Found {len(gt_files)} GT meshes in {args.src_dir}")
    print(f"Strategy: bfs_patch x{args.n_patches}  per-patch ratio "
          f"[{args.ratio_min}, {args.ratio_max}]  -> {args.out_dir}\n")

    # Fixed topology: build adjacency once from the first mesh.
    first = load_mesh(gt_files[0], process=False)
    first_verts = np.array(first.vertices, dtype=np.float32)
    faces_np = np.array(first.faces)
    adj = _build_adjacency(faces_np, len(first_verts))

    # Hop-distance from the torso core (vertex nearest the centroid). Topology is
    # fixed, so this is computed ONCE and shared across all meshes.
    hop = None
    if args.region == 'flat':
        core_vertex = int(np.argmin(
            np.linalg.norm(first_verts - first_verts.mean(axis=0), axis=1)))
        hop = hop_distance(adj, core_vertex)
        print(f"Extremity exclusion: hop-distance from core vertex {core_vertex}, "
              f"keeping <= {args.hop_pct:.0f}th pct (hands/feet/head-top dropped)\n")

    total_pct = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        verts = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)

        # Curvature is per-mesh (pose-dependent), so recompute for each body.
        curv = vertex_curvature(verts, adj) if args.region == 'flat' else None
        mask = make_gentle_mask(verts, faces_np, adj, args.n_patches,
                                (args.ratio_min, args.ratio_max), rng,
                                region=args.region, curv=curv,
                                hop=hop, hop_pct=args.hop_pct)
        ratio = mask.sum() / len(mask)
        total_pct.append(100 * ratio)

        # Copy GT into the new folder so the set is self-contained.
        shutil.copy2(gt_file, os.path.join(args.out_dir, os.path.basename(gt_file)))

        # Partial mesh (missing vertices zeroed, same as the training convention).
        partial = verts.copy()
        partial[mask] = 0.0
        save_ply_explicit(
            Trimesh(vertices=partial, faces=faces_np.copy(), process=False),
            os.path.join(args.out_dir, f'{base}_partial_bfs_patch_r{ratio:.2f}.ply'))

        np.save(os.path.join(args.out_dir, f'{base}_mask.npy'), mask)

        print(f"  {base}: {mask.sum()}/{len(verts)} ({100*ratio:.1f}%) missing")

    print(f"\nDone. {len(gt_files)} gentle samples in {args.out_dir}")
    print(f"Mean missing: {np.mean(total_pct):.1f}%  "
          f"(range {np.min(total_pct):.1f}-{np.max(total_pct):.1f}%)")


if __name__ == "__main__":
    main()
