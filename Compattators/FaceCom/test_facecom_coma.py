"""
Run FaceCom's own completion pipeline on our COMA test set.

The generator is the one train_facecom_coma.py trained on our COMA training
split (checkpoints_coma/checkpoint_decoder.pt). Everything downstream of it is
FaceCom's code, called unmodified:

    utils.completion.facial_mesh_completion(in_file, out_file, config, generator)
      -> rigid_registration()   ICP the partial onto the template
      -> fit()                  optimize z against point-to-surface distance,
                                spherical latent regularization, Adam(0.1),
                                StepLR(10, 0.9), max 1000 iters
      -> post_processing()      threshold the completed region, project it onto
                                the input surface, Laplacian smooth, refine

We supply two things their code needs and the environment does not have:
  * quad_mesh_simplify  -> the identical compiled extension installed as
                           `simplify` in the 3dmesh env (_shim.py)
  * pytorch3d           -> pytorch3d_shim.py, exact point-to-triangle distance
                           plus the containers their pytorch3d_extend.py builds

INPUT. FaceCom takes a real partial mesh -- vertices absent, faces removed,
boundary loops present. Our test set stores partials as full template topology
with missing vertices collapsed to the origin, which is not a mesh their ICP or
their surface distance can consume. So we feed the holed meshes written by
3DGeoMeshNet/make_holed_partials.py, which are the same inputs the
3DGeoMeshNet pipeline row in the paper was scored on.

SCORING. Observed vertices are copied back from ground truth before scoring, so
known-region error is exactly 0. That is the shared protocol every row of the
comparison table is held to (see run_baseline_3dgeomeshnet.py --blend gt).

UNITS. FaceCom's released code is written for millimetre data: post_processing()
flags the completed region with `distance > 4`, and facial_mesh_completion()
passes loss_convergence=1e-7 to fit(). COMA is in metres. --units mm rescales
the meshes so those constants mean what FaceCom intended; --units native runs on
COMA's own scale, where the threshold cannot fire and post_processing() returns
the fit unchanged. Both are reported -- neither is a change to their code.

Usage (conda env: 3dmesh)
  python test_facecom_coma.py --units mm
  python test_facecom_coma.py --units native
"""

import os
import sys
import time
import shutil
import argparse
from glob import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FACECOM_ROOT = os.path.abspath(os.path.join(HERE, "..", "FaceCom_repo"))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, HERE)           # our quad_mesh_simplify.py, the shims
sys.path.insert(1, FACECOM_ROOT)   # must precede 3Dis: it also ships utils/, config/

TEST_DIR = os.path.join(DIS_ROOT, "data", "completion_test_samples", "test_set")
HOLED_DIR = os.path.join(HERE, "..", "3DGeoMeshNet", "holed")


def scaled_workspace(config, scale, work):
    """A copy of config whose template and norm are scaled by `scale`.

    load_norm() reads dataset_dir/norm.pt and fit() applies `out * std + mean`,
    so scaling mean and std is what puts the generator's output in the target
    units. The checkpoint itself is untouched.
    """
    import torch
    import trimesh

    if scale == 1.0:
        return config
    os.makedirs(work, exist_ok=True)

    tm = trimesh.load(config["template_file"], process=False)
    tm.vertices = np.asarray(tm.vertices) * scale
    tpl = os.path.join(work, "template_scaled.ply")
    tm.export(tpl)

    norm = torch.load(os.path.join(config["dataset_dir"], "norm.pt"),
                      weights_only=False)
    torch.save({"mean": norm["mean"] * scale, "std": norm["std"] * scale},
               os.path.join(work, "norm.pt"))

    cfg = dict(config)
    cfg["template_file"] = tpl
    cfg["dataset_dir"] = work
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file",
                    default=os.path.join(HERE, "config_facecom_coma.cfg"))
    ap.add_argument("--units", choices=["mm", "native"], default="mm")
    ap.add_argument("--holed_dir", default=HOLED_DIR)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lambda_reg", type=float, default=None,
                    help="passed through to fit(); FaceCom's face_completion.py "
                         "uses 0.1, omit to use the cfg's value")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import torch
    import trimesh
    import _shim
    import pytorch3d_shim
    _shim.install(FACECOM_ROOT)
    _shim.allow_full_unpickle()
    pytorch3d_shim.install()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        _orig = torch.load          # load_generator hardcodes map_location='cuda'
        torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})

    from config.config import read_config
    config = read_config(args.config_file)

    ckpt = os.path.join(config["checkpoint_dir"], "checkpoint_decoder.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"ABORT: {ckpt} not found. Train first:\n"
                 f"  python train_facecom_coma.py --config_file {args.config_file}")

    holed = sorted(glob(os.path.join(args.holed_dir, "*_holed.ply")))
    if not holed:
        sys.exit(f"ABORT: no *_holed.ply in {args.holed_dir}\n"
                 f"  Run:  cd ..\\3DGeoMeshNet && python make_holed_partials.py")

    scale = 1000.0 if args.units == "mm" else 1.0
    work = os.path.join(HERE, "_scratch_" + args.units)
    config = scaled_workspace(config, scale, work)

    out_dir = args.out_dir or os.path.join(
        FACECOM_ROOT, f"completed_facecom_{args.units}")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(work, "in")
    os.makedirs(tmp_dir, exist_ok=True)

    print("=" * 70)
    print(f"FaceCom pipeline on our COMA test set  [units: {args.units}]")
    print("=" * 70)
    print(f"  decoder   : {ckpt}")
    print(f"  input     : {args.holed_dir}  ({len(holed)} holed meshes)")
    print(f"  output    : {out_dir}")
    print(f"  device    : {device}")
    print("=" * 70)

    from utils.funcs import load_generator
    from utils.completion import facial_mesh_completion
    print("\nbuilding the decimation hierarchy and loading the decoder ...")
    generator = load_generator(config).to(device)

    faces = np.asarray(trimesh.load(os.path.join(DIS_ROOT, "data", "template.ply"),
                                    process=False).faces)

    print(f"\n{'sample':<10} {'missing':>8} {'hole mm':>10} {'full mm':>9} {'s':>7}")
    print("-" * 50)
    rows = []
    for hf in holed:
        base = os.path.basename(hf).replace("_holed.ply", "")
        gt_file = os.path.join(TEST_DIR, "gt", f"{base}_gt_complete.ply")
        mask_file = os.path.join(TEST_DIR, "partial", f"{base}_mask.npy")
        if not (os.path.exists(gt_file) and os.path.exists(mask_file)):
            continue

        # FaceCom reads and writes files; scale in on the way there, out on the
        # way back, so their constants see the units they were written for.
        m = trimesh.load(hf, process=False)
        m.vertices = np.asarray(m.vertices) * scale
        in_file = os.path.join(tmp_dir, f"{base}.ply")
        m.export(in_file)
        out_file = os.path.join(tmp_dir, f"{base}_out.ply")

        t0 = time.time()
        facial_mesh_completion(in_file, out_file, config, generator,
                               lambda_reg=args.lambda_reg,
                               verbose=not args.quiet, rr=True)
        dt = time.time() - t0

        pred = np.asarray(trimesh.load(out_file, process=False).vertices,
                          dtype=np.float64) / scale
        gt = np.asarray(trimesh.load(gt_file, process=False).vertices,
                        dtype=np.float64)
        if pred.shape != gt.shape:
            print(f"{base:<10} skipped: {pred.shape} != {gt.shape}")
            continue

        mask = np.load(mask_file).astype(bool)
        comp = pred.copy()
        comp[~mask] = gt[~mask]                  # shared protocol: known = truth

        trimesh.Trimesh(vertices=comp, faces=faces, process=False).export(
            os.path.join(out_dir, f"{base}_completed_facecom.ply"))

        e = np.linalg.norm(comp - gt, axis=1) * 1000.0
        rows.append((base, mask.mean() * 100, e[mask].mean(), e.mean()))
        print(f"{base:<10} {rows[-1][1]:>7.1f}% {rows[-1][2]:>10.3f} "
              f"{rows[-1][3]:>9.3f} {dt:>7.1f}")

    print("-" * 50)
    if rows:
        h = np.array([r[2] for r in rows])
        print(f"ALL        {np.mean([r[1] for r in rows]):>7.1f}% "
              f"{h.mean():>10.3f} {np.mean([r[3] for r in rows]):>9.3f}")
        print(f"\nmissing-region error: {h.mean():.3f} +/- {h.std(ddof=1):.3f} mm "
              f"over {len(h)} samples")
    print(f"\nwrote {out_dir}")
    print("breakdown by mask type:\n"
          f"  python ..\\breakdown_by_mask.py --completed_dir {out_dir} "
          f"--suffix _completed_facecom.ply --method FaceCom")


if __name__ == "__main__":
    main()
