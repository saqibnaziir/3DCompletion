"""
Measure what it costs to train FaceCom's generator at COMA resolution.

FaceCom's released config is sized for its own 10,857-vertex template. At COMA's
5,023 vertices the parameter count and memory change, so rather than guess at a
training time we build the real model, run real training steps, and extrapolate.

Reports, per batch size: peak GPU memory and seconds/step, then the projected
wall-clock for 300 epochs over our 16,368-mesh training split.

Usage (conda env: 3dmesh)
  python bench_facecom_coma.py
  python bench_facecom_coma.py --batches 1 2 4 8 --steps 12
"""

import os
import sys
import time
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FACECOM_ROOT = os.path.abspath(os.path.join(HERE, "..", "FaceCom_repo"))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

import _shim
_shim.install(FACECOM_ROOT)

N_TRAIN = 16368          # our COMA training split
N_VERT = 5023


def build(config):
    from utils.funcs import get_mesh_matrices
    from utils.models import FMGenModel
    pA, pD, pU = get_mesh_matrices(config)
    return FMGenModel(config, pA, pD, pU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"        {p.name}, {p.total_memory/1e9:.1f} GB")

    # FaceCom's released model parameters, on OUR template.
    config = {
        "template_file": os.path.join(DIS_ROOT, "data", "template.ply"),
        "n_layers": 3,
        "z_length": 256,
        "down_sampling_factors": [4, 4, 4],
        "num_features_local": [3, 8, 16, 32],
        "num_features_global": [3, 64, 512, 2048],
        "batch_norm": True,
        "lambda_reg": 0.001,
    }

    print("\nbuilding model at COMA resolution ...")
    model = build(config).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"  encoder {n_enc/1e6:7.2f}M   decoder {n_dec/1e6:7.2f}M   "
          f"total {(n_enc+n_dec)/1e6:.2f}M parameters")

    from utils.funcs import get_edge_index
    from trimesh import load_mesh
    tm = load_mesh(config["template_file"])
    edge_index = get_edge_index(tm.vertices, tm.faces).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    import torch.nn.functional as F

    print(f"\n{'batch':>6} {'peak GB':>9} {'s/step':>9} {'steps/epoch':>12} "
          f"{'min/epoch':>10} {'h for '+str(args.epochs)+'ep':>13}")
    print("-" * 64)
    for bs in args.batches:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs * N_VERT, 3, device=device)

            class _B:  # minimal stand-in for a PyG Batch
                pass
            batch = _B()
            batch.x = x
            batch.num_graphs = bs
            batch.edge_index = edge_index
            batch.y = x

            model.train()
            for _ in range(3):                     # warm-up
                opt.zero_grad()
                out, z = model(batch)
                F.mse_loss(out, batch.y).backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(args.steps):
                opt.zero_grad()
                out, z = model(batch)
                loss = F.mse_loss(out, batch.y) + F.l1_loss(out, batch.y)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / args.steps

            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
            spe = N_TRAIN / bs
            mpe = spe * dt / 60
            print(f"{bs:>6} {peak:>9.2f} {dt:>9.3f} {spe:>12.0f} {mpe:>10.1f} "
                  f"{mpe*args.epochs/60:>13.1f}")
        except RuntimeError as e:
            msg = "OUT OF MEMORY" if "out of memory" in str(e).lower() else str(e)[:40]
            print(f"{bs:>6} {msg:>50}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
