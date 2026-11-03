"""
Convert our test partials into GENUINE holed meshes for 3DGeoMeshNet's process.py.

Our test set stores a partial as full template topology (5023 verts, 9976 faces)
with the missing vertices collapsed to the origin and every face still present.
3DGeoMeshNet's front-end (Complete/oliver/7th/COMA/process.py) instead expects a
real partial scan: missing vertices absent, faces touching them removed, so the
mesh has actual boundary loops and connected components to match against the
template.

This writes those holed meshes, plus a per-sample .npz recording which template
vertex each surviving vertex came from -- that index map is NOT given to
process.py (it has to recover correspondence itself, which is the point), it is
only used afterwards to audit how well its ICP + Hungarian matching recovered
the true correspondence.

Usage (conda env: 3dmesh)
  python make_holed_partials.py
  -> holed/test_XXX_holed.ply  and  holed/test_XXX_index.npz
"""

import os
import sys
from glob import glob

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TEST_DIR = os.path.join(DIS_ROOT, "data", "completion_test_samples", "test_set")


def main():
    out_dir = os.path.join(HERE, "holed")
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(TEST_DIR, "gt", "*_gt_complete.ply")))
    if not gt_files:
        sys.exit(f"No GT meshes under {TEST_DIR}/gt")

    tmpl = trimesh.load(os.path.join(DIS_ROOT, "data", "template.ply"), process=False)
    faces = np.asarray(tmpl.faces)

    print(f"{'sample':<10} {'missing%':>9} {'verts kept':>11} {'faces kept':>11} {'components':>11}")
    print("-" * 58)
    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace("_gt_complete.ply", "")
        mask = np.load(os.path.join(TEST_DIR, "partial", f"{base}_mask.npy")).astype(bool)
        obs = ~mask                                    # True = observed
        gt = np.asarray(trimesh.load(gt_file, process=False).vertices)

        keep_idx = np.flatnonzero(obs)                 # template indices we keep
        remap = -np.ones(len(obs), dtype=np.int64)
        remap[keep_idx] = np.arange(len(keep_idx))

        # keep only faces whose three corners all survive
        fmask = obs[faces].all(axis=1)
        new_faces = remap[faces[fmask]]

        holed = trimesh.Trimesh(vertices=gt[keep_idx], faces=new_faces, process=False)
        # drop vertices orphaned by face removal, and keep the index map in sync
        used = np.zeros(len(keep_idx), dtype=bool)
        used[np.unique(new_faces)] = True
        if not used.all():
            reremap = -np.ones(len(keep_idx), dtype=np.int64)
            reremap[np.flatnonzero(used)] = np.arange(int(used.sum()))
            holed = trimesh.Trimesh(vertices=holed.vertices[used],
                                    faces=reremap[new_faces], process=False)
            keep_idx = keep_idx[used]

        n_cc = len(holed.split(only_watertight=False))
        holed.export(os.path.join(out_dir, f"{base}_holed.ply"))
        np.savez(os.path.join(out_dir, f"{base}_index.npz"),
                 template_index=keep_idx, mask=mask)

        print(f"{base:<10} {100*mask.mean():>8.1f}% {len(holed.vertices):>11} "
              f"{len(holed.faces):>11} {n_cc:>11}")

    print(f"\nwrote {len(gt_files)} holed meshes -> {out_dir}")


if __name__ == "__main__":
    main()
