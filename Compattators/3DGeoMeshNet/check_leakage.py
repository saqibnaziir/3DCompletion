"""
Does the 3DGeoMeshNet COMA checkpoint's training set contain OUR test meshes?

This decides whether its completion numbers measure generalization or
memorization. Our 20 test meshes come from the 3Dis custom 80/20 pooled-frame
COMA split; 3DGeoMeshNet used its own split, so overlap is likely and would
make its scores an upper bound that no honest baseline row can quote.

Method: load E:'s processed training/eval tensors, denormalize back to world
coordinates with the matching norm.pt, and for each of our test GT meshes find
the nearest training mesh under mean per-vertex L2. A distance at numerical zero
means that exact frame was trained on.

Usage (conda env: 3dmesh)
  python check_leakage.py                # training.pt
  python check_leakage.py --split eval   # eval.pt
"""

import os
import argparse

import numpy as np
import torch
import trimesh
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
E_DATA = r"E:\PosDoc\COSURAI\3DGEOMeshNET\data"
N_VERT = 5023


def load_split(path):
    """Return (M, N_VERT, 3) of normalized vertices from a PyG InMemoryDataset."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    data = obj[0] if isinstance(obj, tuple) else obj
    x = getattr(data, "x", None)
    if x is None and isinstance(data, dict):
        x = data.get("x")
    if x is None:
        raise SystemExit(f"could not find .x in {path}; got {type(data)}")
    x = x.float().numpy()
    if x.shape[0] % N_VERT:
        raise SystemExit(f"{path}: {x.shape[0]} rows is not a multiple of {N_VERT}")
    return x.reshape(-1, N_VERT, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "eval"], default="training")
    ap.add_argument("--norm", default=os.path.join(E_DATA, "norm.pt"))
    args = ap.parse_args()

    path = os.path.join(E_DATA, "processed", f"{args.split}.pt")
    print(f"loading {path} ({os.path.getsize(path)/1e6:.0f} MB) ...")
    X = load_split(path)
    print(f"  {X.shape[0]} meshes of {X.shape[1]} vertices")

    n = torch.load(args.norm, map_location="cpu", weights_only=False)
    mean, std = n["mean"].numpy(), n["std"].numpy()

    # The stored tensors may be normalized or raw; decide by magnitude.
    if np.abs(X[0]).max() > 1.0:
        print("  stored RAW (world coordinates)")
        W = X
    else:
        print("  stored NORMALIZED -> denormalizing with norm.pt")
        W = X * std + mean

    gt_files = sorted(glob(os.path.join(
        DIS_ROOT, r"data\completion_test_samples\test_set\gt", "*_gt_complete.ply")))

    # Cheap signature first (per-mesh vertex mean), then exact check on the best.
    sig_tr = W.mean(axis=1)                              # (M,3)
    print(f"\n{'test mesh':<10} {'nearest train idx':>18} {'mean L2 (mm)':>14}   verdict")
    print("-" * 66)
    hits = 0
    for f in gt_files:
        base = os.path.basename(f).replace("_gt_complete.ply", "")
        g = np.asarray(trimesh.load(f, process=False).vertices, dtype=np.float32)
        d0 = np.linalg.norm(sig_tr - g.mean(0), axis=1)
        cand = np.argsort(d0)[:64]                       # rescore the closest 64 exactly
        d = np.array([np.linalg.norm(W[i] - g, axis=1).mean() for i in cand])
        j = int(np.argmin(d))
        mm = 1000.0 * float(d[j])
        leaked = mm < 0.01
        hits += leaked
        print(f"{base:<10} {int(cand[j]):>18} {mm:>14.4f}   "
              f"{'*** IN TRAINING SET ***' if leaked else 'not present'}")

    print("-" * 66)
    print(f"{hits}/{len(gt_files)} of our test meshes are in E:'s {args.split} split")
    if hits:
        print("\n  Its completion scores are memorization, not generalization, and\n"
              "  cannot be quoted as a baseline without disclosing this.")


if __name__ == "__main__":
    main()
