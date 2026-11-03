"""
Regenerate test partial meshes + masks with a new mask ratio range.
Reuses the existing GT complete meshes, just creates new partial + mask files.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from glob import glob
from trimesh import load_mesh, Trimesh
from utils.funcs import save_ply_explicit
from utils.masking import generate_random_mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--mask_min", type=float, default=0.05)
    parser.add_argument("--mask_max", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    test_dir = args.test_dir
    rng = np.random.RandomState(args.seed)

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    print(f"Found {len(gt_files)} GT meshes in {test_dir}")
    print(f"Mask ratio range: [{args.mask_min}, {args.mask_max}]")

    # We need face connectivity — load from the first GT mesh
    first_mesh = load_mesh(gt_files[0], process=False)
    faces_np = np.array(first_mesh.faces)

    # Remove old partial and mask files
    old_partials = glob(os.path.join(test_dir, '*_partial_*.ply'))
    old_masks = glob(os.path.join(test_dir, '*_mask.npy'))
    for f in old_partials + old_masks:
        os.remove(f)
        print(f"  Removed: {os.path.basename(f)}")

    # Generate new ones
    for gt_file in gt_files:
        basename = os.path.basename(gt_file).replace('_gt_complete.ply', '')

        gt_mesh = load_mesh(gt_file, process=False)
        verts_np = np.array(gt_mesh.vertices, dtype=np.float32)

        mask, strategy, ratio = generate_random_mask(
            verts_np, faces_np,
            mask_ratio_range=(args.mask_min, args.mask_max),
            rng=rng
        )

        # Save partial mesh
        partial_verts = verts_np.copy()
        partial_verts[mask] = 0.0
        partial_mesh = Trimesh(vertices=partial_verts, faces=faces_np.copy(), process=False)
        partial_path = os.path.join(test_dir, f'{basename}_partial_{strategy}_r{ratio:.2f}.ply')
        save_ply_explicit(partial_mesh, partial_path)

        # Save mask
        mask_path = os.path.join(test_dir, f'{basename}_mask.npy')
        np.save(mask_path, mask)

        num_missing = mask.sum()
        print(f"  {basename}: {strategy} {num_missing}/{len(verts_np)} "
              f"({100*num_missing/len(verts_np):.1f}%) missing")

    print("\nDone. New partial meshes and masks generated.")


if __name__ == "__main__":
    main()
