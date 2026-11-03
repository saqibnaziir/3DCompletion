"""
Stage OUR exact COMA training split as a FaceCom dataset directory.

Why not just point FaceCom at data/train:
  1. FaceCom's MyDataset globs every mesh under <dataset_dir>/train and makes its
     OWN random 80/20 split. Pointed at all of COMA it would train on ~80% of our
     20 test meshes -- straight leakage, and the resulting baseline number would
     be meaningless.
  2. Our split cannot be regenerated. utils/completion_dataset.py calls
     random.shuffle(indices) with no seed set (the comment claims otherwise), so
     re-running process() yields a different partition. The only surviving record
     of the split actually used is data/processed_completion/training.pt.

So we export the training meshes straight out of that tensor. FaceCom then does
its internal 80/20 on OUR training meshes only, and our test meshes never enter
its pipeline at any point.

This also verifies the held-out property directly: every test GT mesh is checked
for membership in training.pt.

Usage (conda env: 3dmesh)
  python stage_coma_trainset.py --verify_only
  python stage_coma_trainset.py
"""

import os
import argparse
from glob import glob

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROCESSED = os.path.join(DIS_ROOT, "data", "processed_completion")
TEST_DIR = os.path.join(DIS_ROOT, "data", "completion_test_samples", "test_set")
N_VERT = 5023


def load_split(name):
    """Return (M, N_VERT, 3) WORLD-coordinate vertices from a PyG collate.

    The completion dataset stores the mean-subtracted offset field, not world
    coordinates (norm.pt has std=None, so normalization is mean subtraction
    only and the offset field equals the normalized vertices). The stored
    tensor must therefore have the mean added back before it can be compared
    against, or exported alongside, ground-truth meshes.
    """
    path = os.path.join(PROCESSED, f"{name}.pt")
    data, slices = torch.load(path, map_location="cpu", weights_only=False)
    x = data.x.float().numpy()
    if x.shape[0] % N_VERT:
        raise SystemExit(f"{path}: {x.shape[0]} rows not a multiple of {N_VERT}")
    x = x.reshape(-1, N_VERT, 3)

    norm = torch.load(os.path.join(PROCESSED, "norm.pt"),
                      map_location="cpu", weights_only=False)
    mean = norm["mean"].float().numpy()
    std = norm["std"]
    if std is not None:
        x = x * std.float().numpy()
    return x + mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(HERE, "coma_dataset"))
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    print("loading our split ...")
    tr = load_split("training")
    ev = load_split("eval")
    print(f"  training.pt : {tr.shape[0]} meshes")
    print(f"  eval.pt     : {ev.shape[0]} meshes")
    print(f"  total       : {tr.shape[0] + ev.shape[0]}")
    # Sanity: reconstructed world meshes must span the template's extent
    # (~0.19 m), not the offset field's (~0.02 m).
    span = float(np.linalg.norm(np.ptp(tr[0], axis=0)))
    print(f"  reconstructed world coords: bbox diag {span:.4f} m "
          f"({'OK' if span > 0.3 else 'WRONG FRAME'})")

    # --- held-out verification -------------------------------------------
    gt_files = sorted(glob(os.path.join(TEST_DIR, "gt", "*_gt_complete.ply")))
    sig_tr = tr.mean(axis=1)
    print(f"\nverifying {len(gt_files)} test meshes are held out of training.pt")
    print(f"  {'test mesh':<10} {'nearest train (mm)':>20}   verdict")
    print("  " + "-" * 52)
    leaked = 0
    for f in gt_files:
        base = os.path.basename(f).replace("_gt_complete.ply", "")
        g = np.asarray(trimesh.load(f, process=False).vertices, dtype=np.float32)
        cand = np.argsort(np.linalg.norm(sig_tr - g.mean(0), axis=1))[:64]
        d = min(np.linalg.norm(tr[i] - g, axis=1).mean() for i in cand)
        mm = 1000.0 * float(d)
        bad = mm < 0.01
        leaked += bad
        print(f"  {base:<10} {mm:>20.4f}   {'*** LEAKED ***' if bad else 'held out'}")
    print("  " + "-" * 52)
    print(f"  {leaked}/{len(gt_files)} leaked")
    if leaked:
        raise SystemExit("ABORT: test meshes are in the training split.")

    if args.verify_only:
        print("\nverify_only: nothing written.")
        return

    # --- export ------------------------------------------------------------
    train_dir = os.path.join(args.out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    faces = trimesh.load(os.path.join(DIS_ROOT, "data", "template.ply"),
                         process=False).faces
    print(f"\nwriting {tr.shape[0]} meshes -> {train_dir}")
    for i in range(tr.shape[0]):
        trimesh.Trimesh(tr[i].astype(np.float64), faces, process=False).export(
            os.path.join(train_dir, f"coma_train_{i:05d}.ply"))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{tr.shape[0]}")
    print(f"done. point FaceCom's dataset_dir at {args.out_dir}")


if __name__ == "__main__":
    main()
