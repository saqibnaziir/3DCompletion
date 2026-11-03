"""Decisive validation: OVERFIT a tiny subset, then sample.
If standardization + DDIM sampling are correct, an overfit model MUST produce
output at data scale (~0.05 absmax), not an explosion."""
import os, sys; sys.path.append(os.path.abspath('.'))
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion
from utils.diffusion_model import GraphUNetDenoiser
from scripts.train_diffusion import parse_diffusion_section


def main():
    CFG = "config/config_diffusion.cfg"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = read_config(CFG); dcfg = parse_diffusion_section(CFG)

    ds = MeshCompletionDataset(cfg, 'train')
    N = ds.get(0).x.shape[0]
    sigma = ds.data.y.std(0).clamp(min=1e-6).to(device)
    print("sigma:", sigma.tolist())
    data_absmax = float(ds.data.y.abs().max())
    std_absmax = float((ds.data.y.to(device) / sigma).abs().max())
    clip_value = std_absmax * 1.1                      # generous bound, won't clip real data
    print(f"data offset absmax={data_absmax:.4f}  std={float(ds.data.y.std()):.5f}")
    print(f"standardized absmax={std_absmax:.2f}  -> clip_value={clip_value:.1f}")

    sub = Subset(ds, list(range(16)))                 # overfit just 16 meshes
    loader = DataLoader(sub, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

    pA, pD, pU = get_mesh_matrices(cfg)
    den = GraphUNetDenoiser(list(pA), list(pD), list(pU), 7, 3,
                            dcfg['base_channels'], dcfg['channel_mults'],
                            dcfg['time_embed_dim']).to(device)
    diff = GaussianDiffusion(dcfg['timesteps'], dcfg['schedule']).to(device)
    opt = torch.optim.Adam(den.parameters(), lr=5e-4)

    print("overfitting 16 meshes for 5000 steps...")
    den.train(); it = iter(loader); running = 0.0
    for step in range(5000):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        batch = batch.to(device)
        x0 = batch.y / sigma
        cond = batch.x.clone(); cond[:, :3] = cond[:, :3] / sigma
        loss, *_ = diff.p_losses(den, x0, cond, batch.num_graphs, N, batch.vertex_mask, 1.0, 3.0)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1}: loss {running/1000:.4f}"); running = 0.0

    den.eval()
    d = ds.get(0).to(device)
    cond1 = d.x.clone(); cond1[:, :3] = cond1[:, :3] / sigma
    known = cond1[:, 3] > 0.5; kv = cond1[:, :3].contiguous(); miss = ~known
    x0_true = (d.y / sigma)                            # standardized target

    # --- DIAGNOSTIC 1: x0 recovery at fixed t (no sampling instability) ---
    # At low t, sqrt(acp)~1 so x0 recovery is stable -> pure measure of learning.
    print("\n--- x0-recovery on MISSING region (standardized space) ---")
    for tval in [20, 100, 400]:
        t_node = torch.full((N,), tval, device=device, dtype=torch.long)
        noise = torch.randn_like(x0_true)
        x_t = diff.q_sample(x0_true, t_node, noise)
        with torch.no_grad():
            eps = den(x_t, cond1, torch.tensor([tval], device=device), 1)
        x0_rec = diff.predict_x0_from_eps(x_t, t_node, eps)
        rec_err = (x0_rec[miss] - x0_true[miss]).norm(dim=1).mean().item()
        print(f"  t={tval:4d}: x0 recovery err={rec_err:.4f}  "
              f"(target std~1; absmax_pred={float(x0_rec[miss].abs().max()):.2f})")

    # --- DIAGNOSTIC 2: full DDIM sampling WITH clamping ---
    with torch.no_grad():
        x0s = diff.ddim_sample(den, cond1, 1, N, known, kv, num_steps=50,
                               device=device, clip_value=clip_value)
    offset = x0s * sigma
    err = (offset[miss] - d.y[miss]).norm(dim=1).mean().item()
    print("\n--- clamped DDIM sample, MISSING region (unscaled offset space) ---")
    print(f"  sampled absmax : {float(offset[miss].abs().max()):.5f}  (data ~{data_absmax:.3f})")
    print(f"  per-vertex err : {err:.5f}  (vs baseline ~0.0049)")
    ok = float(offset[miss].abs().max()) < 0.2
    print("\nSCALE/PIPELINE OK" if ok else "\nSTILL OUT OF SCALE")


if __name__ == "__main__":
    main()
