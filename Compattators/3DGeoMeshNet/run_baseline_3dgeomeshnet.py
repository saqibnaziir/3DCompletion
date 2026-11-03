"""
Baseline adapter: 3DGeoMeshNet (multi-scale graph auto-encoder) -> our test set.

3DGeoMeshNet completes a partial mesh in two stages:
  STAGE 1 (front-end)  Complete/oliver/7th/COMA/process.py
                       ICP to the template, split into connected components,
                       Hungarian match, fill unmatched slots with template
                       vertices -> a complete mesh in template topology.
  STAGE 2 (network)    the graph auto-encoder reconstructs that mesh, so the
                       template-filled hole is replaced by what the learned
                       shape prior predicts.
This script runs STAGE 2 on STAGE 1's output and writes meshes scorable by
scripts/eval_completion_dir.py.

INPUT MODES
  --input_dir <dir>   faithful: consume process.py output (run make_holed_partials.py
                      then process.py --batch first). Correspondence is whatever
                      the front-end recovered, errors included.
  (omit)              oracle: build the stage-1 input directly from our mask
                      (observed = truth, missing = mean shape). Grants perfect
                      registration for free -- GENEROUS to the baseline, and it
                      isolates the completion model from the front-end.

BLENDING (how the observed region is treated in the output)
  --blend gt          copy OUR ground-truth observed vertices back verbatim, so
                      known-region error is exactly 0 -- the shared protocol
                      every method in the table is held to. DEFAULT.
  --blend pipeline    keep the front-end's own observed vertices, so the number
                      includes its registration error. The honest "full system"
                      cost; not directly comparable to the main table.

NETWORK MODES
  --mode ae           single encode->decode pass.
  --mode fit          initialize z from the encoder, then optimize it against
                      the OBSERVED vertices with a ||z||^2 prior (Adam,
                      --iters steps), then decode.

CHECKPOINTS -- READ THIS
  The pair in D:\\COURSAI\\3DGeoMeshNet\\data was trained on a SINGLE COMA
  subject (data/train holds only FaceTalk_170725_00137_TA; its norm.pt has
  median per-vertex std 0.000867 against 0.002015 for every other COMA norm.pt
  in that repo, ours included). Our multi-subject test meshes land at ~6 sigma
  under it and it returns 18-44 mm garbage. Point --checkpoint_dir/--norm_file
  at multi-subject COMA weights instead. The norm gate below refuses to emit
  numbers when the normalization does not fit; pass --no_strict_norm to override
  deliberately.

USAGE (conda env: 3dmesh)
  python make_holed_partials.py
  cd D:\\COURSAI\\3DGeoMeshNet\\Complete\\oliver\\7th\\COMA
  python process.py -template template.ply -input <...>/holed -output <...>/proc_out --batch
  cd back
  python run_baseline_3dgeomeshnet.py --input_dir proc_out --mode fit \
      --checkpoint_dir <dir with checkpoint_encoder.pt/checkpoint_decoder.pt> \
      --norm_file <matching norm.pt>

  python ../../scripts/eval_completion_dir.py \
      --completed_dir completed_fit --suffix _completed_3dgeomeshnet.ply \
      --method 3DGeoMeshNet --out_csv ../../data/completion_test_samples/baseline_results.csv
"""

import os
import sys
import csv
import time
import argparse
from glob import glob

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo holding the 3DGeoMeshNet source.
#
# The E: copy sits next to the full-COMA checkpoints, but its utils/funcs.py
# does `from quad_mesh_simplify import simplify_mesh`, which is not installed in
# the 3dmesh env. The D: copy is the SAME code with that one import swapped for
# a vendored `simplify` package -- generate_transform_matrices_trimesh,
# generate_VFDU and get_edge_index are identical in both, so the mesh hierarchy
# the conv weights were trained against is reproduced exactly. Override with
# --repo_root if you install quad_mesh_simplify.
DEFAULT_REPO_ROOT = r"D:\COURSAI\3DGeoMeshNet"
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_TEST_DIR = os.path.join(DIS_ROOT, "data", "completion_test_samples", "test_set")


class _ZeroResidual(torch.nn.Module):
    """Stands in for a residual projection that the trained model did not have.

    Adding 0 reproduces the legacy forward pass `x_local = conv(x_local)`
    exactly. Identity() would be wrong twice over: it would inject a residual
    the trained weights never saw, and the local path changes channel count at
    every level (32->16->8->3) so identity is not even shape-valid.
    """

    def forward(self, x):
        return 0.0


def load_generator_on(config, device):
    """Mirror of 3DGeoMeshNet's utils.funcs.load_generator, device-aware, and
    tolerant of the legacy decoder layout."""
    from utils.funcs import get_mesh_matrices
    from utils.models import FMGenModel

    A, D, U = get_mesh_matrices(config)
    model = FMGenModel(config, A, D, U)

    enc_sd = torch.load(os.path.join(config["checkpoint_dir"], "checkpoint_encoder.pt"),
                        map_location=device)
    dec_sd = torch.load(os.path.join(config["checkpoint_dir"], "checkpoint_decoder.pt"),
                        map_location=device)
    model.encoder.load_state_dict(enc_sd)

    # Checkpoints saved before local_residual_projections was added to the
    # decoder have 32 tensors where current code builds 38.
    legacy = not any("local_residual_projections" in k for k in dec_sd)
    if legacy and getattr(model.decoder, "local_residual_projections", None) is not None:
        n = len(model.decoder.local_residual_projections)
        model.decoder.local_residual_projections = torch.nn.ModuleList(
            [_ZeroResidual() for _ in range(n)])
        print(f"  legacy decoder detected: neutralized {n} local_residual_projections")
    model.decoder.load_state_dict(dec_sd)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


def reconstruction_check(model, mean, std, gt_files, device, n):
    """Sanity gate: reconstruct COMPLETE meshes (no hole at all).

    A correctly loaded, trained auto-encoder reproduces a COMA face to roughly
    a millimetre. Garbage here means the weights, the normalization or the
    architecture are mismatched, and every completion number would be void.
    """
    errs = []
    for f in gt_files[:n]:
        v = np.asarray(trimesh.load(f, process=False).vertices, dtype=np.float64)
        x = torch.tensor(v, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model.decoder(model.encoder((x - mean) / std, 1), 1) * std + mean
        errs.append(1000.0 * np.linalg.norm(
            v - out.cpu().numpy().astype(np.float64), axis=1).mean())
    errs = np.array(errs)
    print(f"recon check : {errs.mean():.3f} +/- {errs.std():.3f} mm over {len(errs)} "
          f"COMPLETE meshes  (expect ~1 mm for a trained AE)")
    return errs.mean()


def complete_one(x_in_world, gt, mask, model, mean, std,
                 mode, iters, lr, lambda_reg, blend, verbose):
    """Complete one mesh.

    x_in_world : (N,3) stage-1 input in world coordinates
    gt, mask   : ground truth and the missing mask (True = MISSING)
    """
    obs = ~mask
    x_in = (x_in_world - mean) / std

    with torch.no_grad():
        z = model.encoder(x_in, 1)

    if mode == "fit":
        z = z.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.8, patience=50)
        # Fit to the stage-1 input on the OBSERVED vertices only; the hole is
        # never supervised, it is whatever the shape prior extrapolates.
        target = x_in[obs].detach()
        best_loss, best_z = float("inf"), z.clone().detach()

        for it in range(iters):
            opt.zero_grad()
            pred = model.decoder(z, 1)
            data = torch.mean(torch.sum((pred[obs] - target) ** 2, dim=1))
            reg = lambda_reg * torch.mean(z ** 2)
            loss = data + reg
            if not torch.isfinite(loss):
                print(f"    warning: non-finite loss at iter {it}, stopping")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], 1.0)
            opt.step()
            sched.step(loss)
            if loss.item() < best_loss:
                best_loss, best_z = loss.item(), z.clone().detach()
            if verbose and (it + 1) % 250 == 0:
                print(f"    iter {it+1:5d}  loss {loss.item():.6f}  data {data.item():.6f}")
        z = best_z

    with torch.no_grad():
        out = model.decoder(z, 1) * std + mean

    # Preserve the measurement, matching the shared protocol.
    out[obs] = gt[obs] if blend == "gt" else x_in_world[obs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file",
                    default=os.path.join(HERE, "config_3dgeomeshnet_coma_E.cfg"))
    ap.add_argument("--repo_root", default=DEFAULT_REPO_ROOT,
                    help="3DGeoMeshNet source tree to import the architecture from")
    ap.add_argument("--recon_check", type=int, default=5,
                    help="reconstruct N complete meshes as a load sanity gate; 0 to skip")
    ap.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--input_dir", default=None,
                    help="dir of process.py output; omit for oracle-correspondence input")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--checkpoint_dir", default=None, help="overrides the config")
    ap.add_argument("--norm_file", default=None,
                    help="norm.pt matching the checkpoint; default <dataset_dir>/norm.pt")
    ap.add_argument("--mode", choices=["ae", "fit"], default="fit")
    ap.add_argument("--blend", choices=["gt", "pipeline"], default="gt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda_reg", type=float, default=0.001)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_strict_norm", action="store_true",
                    help="emit numbers even when the normalization gate fails")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tag = args.mode + ("_pipeline" if args.input_dir else "_oracle")
    out_dir = args.out_dir or os.path.join(HERE, f"completed_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    # The chosen repo must win the import race: 3Dis also has utils/ and config/
    # packages, and importing the wrong one silently loads a different network.
    sys.path.insert(0, args.repo_root)
    print(f"repo        : {args.repo_root}")

    from config.config import read_config
    config = read_config(args.config_file)
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    norm_file = args.norm_file or os.path.join(config["dataset_dir"], "norm.pt")

    print(f"checkpoints : {config['checkpoint_dir']}")
    print(f"norm        : {norm_file}")
    print(f"template    : {config['template_file']}")
    print(f"stage-1 in  : {args.input_dir or '(oracle: mask + mean shape)'}")
    print(f"device      : {device} | mode {args.mode} | blend {args.blend}"
          + (f" | iters {args.iters}" if args.mode == "fit" else ""))

    model = load_generator_on(config, device)
    print(f"generator   : {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    norm = torch.load(norm_file, map_location="cpu", weights_only=False)
    mean = norm["mean"].to(device).float()
    std = norm["std"].to(device).float()
    std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)

    faces = trimesh.load(os.path.join(DIS_ROOT, "data", "template.ply"),
                         process=False).faces

    gt_dir = os.path.join(args.test_dir, "gt")
    gt_dir = gt_dir if os.path.isdir(gt_dir) else args.test_dir
    mask_dir = os.path.join(args.test_dir, "partial")
    mask_dir = mask_dir if os.path.isdir(mask_dir) else args.test_dir

    gt_files = sorted(glob(os.path.join(gt_dir, "*_gt_complete.ply")))
    if not gt_files:
        sys.exit(f"No *_gt_complete.ply under {gt_dir}")
    if args.limit:
        gt_files = gt_files[:args.limit]

    # --- normalization gate -------------------------------------------------
    # If the checkpoint's training statistics do not cover our meshes, the
    # encoder is being fed far-out-of-distribution input and every number below
    # is meaningless. Catch that here rather than in the results table.
    g0 = torch.tensor(np.asarray(trimesh.load(gt_files[0], process=False).vertices),
                      dtype=torch.float32, device=device)
    z0 = (g0 - mean) / std
    zmean, zstd = float(z0.mean()), float(z0.std())
    ok = abs(zmean) < 0.5 and 0.5 < zstd < 2.0
    print(f"norm gate   : normalized GT mean {zmean:+.3f}, std {zstd:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}  (want ~0 and ~1)")
    if not ok:
        msg = ("\n  The checkpoint's normalization does not fit these meshes.\n"
               "  Almost always this means the weights were trained on a much\n"
               "  narrower set than the test data (see the header of this file).\n"
               "  Point --checkpoint_dir/--norm_file at multi-subject COMA weights,\n"
               "  or pass --no_strict_norm to proceed anyway.")
        if not args.no_strict_norm:
            sys.exit("ABORT:" + msg)
        print("WARNING:" + msg)

    if args.recon_check:
        reconstruction_check(model, mean, std, gt_files, device, args.recon_check)

    rows = []
    for k, gt_file in enumerate(gt_files):
        base = os.path.basename(gt_file).replace("_gt_complete.ply", "")
        mask_file = os.path.join(mask_dir, f"{base}_mask.npy")
        if not os.path.exists(mask_file):
            print(f"  skip {base}: no mask"); continue

        gt_np = np.asarray(trimesh.load(gt_file, process=False).vertices, dtype=np.float64)
        mask_np = np.load(mask_file).astype(bool)          # True = MISSING

        if args.input_dir:
            hits = sorted(glob(os.path.join(args.input_dir, f"{base}*.ply")))
            if not hits:
                print(f"  skip {base}: no stage-1 mesh in {args.input_dir}"); continue
            x_np = np.asarray(trimesh.load(hits[0], process=False).vertices, dtype=np.float64)
            if x_np.shape != gt_np.shape:
                print(f"  skip {base}: stage-1 mesh is {x_np.shape}, not template topology")
                continue
        else:
            # observed = truth, missing = mean shape (what stage 1 produces
            # under perfect correspondence)
            x_np = gt_np.copy()
            x_np[mask_np] = np.asarray(mean.cpu(), dtype=np.float64)[mask_np]

        gt = torch.tensor(gt_np, dtype=torch.float32, device=device)
        x_in = torch.tensor(x_np, dtype=torch.float32, device=device)
        mask = torch.tensor(mask_np, device=device)

        print(f"[{k+1:>2}/{len(gt_files)}] {base}  missing {mask_np.mean()*100:.1f}%")
        t0 = time.time()
        out = complete_one(x_in, gt, mask, model, mean, std, args.mode,
                           args.iters, args.lr, args.lambda_reg, args.blend,
                           not args.quiet)
        dt = time.time() - t0

        out_np = out.detach().cpu().numpy().astype(np.float64)
        err = np.linalg.norm(gt_np - out_np, axis=1)
        rows.append({"sample": base,
                     "missing_pct": 100.0 * mask_np.mean(),
                     "missing_mean_mm": 1000.0 * err[mask_np].mean(),
                     "overall_mean_mm": 1000.0 * err.mean(),
                     "known_mean_mm": 1000.0 * err[~mask_np].mean(),
                     "seconds": dt})
        print(f"        hole {rows[-1]['missing_mean_mm']:7.3f} mm | "
              f"full {rows[-1]['overall_mean_mm']:6.3f} mm | "
              f"known {rows[-1]['known_mean_mm']:.3e} mm | {dt:5.1f}s")

        trimesh.Trimesh(out_np, faces, process=False).export(
            os.path.join(out_dir, f"{base}_completed_3dgeomeshnet.ply"))

    if not rows:
        sys.exit("Nothing completed.")

    miss = np.array([r["missing_mean_mm"] for r in rows])
    over = np.array([r["overall_mean_mm"] for r in rows])
    known = np.array([r["known_mean_mm"] for r in rows])
    secs = np.array([r["seconds"] for r in rows])
    print("\n" + "=" * 68)
    print(f"3DGeoMeshNet [{tag}, blend={args.blend}]   {len(rows)} samples")
    print("=" * 68)
    print(f"  hole error   : {miss.mean():.3f} +/- {miss.std():.3f} mm")
    print(f"  full surface : {over.mean():.3f} +/- {over.std():.3f} mm")
    print(f"  known region : {known.mean():.3e} mm")
    print(f"  runtime      : {secs.mean():.1f} s / mesh")
    print(f"  meshes -> {out_dir}")

    csv_path = os.path.join(HERE, f"per_sample_{tag}_{args.blend}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  per-sample CSV -> {csv_path}")


if __name__ == "__main__":
    main()
