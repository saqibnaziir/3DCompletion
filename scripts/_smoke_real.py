"""Real-data smoke test: actual template, D/U matrices, dataset, one batch + one DDIM sample."""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time, torch
from torch_geometric.loader import DataLoader
from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion
from utils.diffusion_model import GraphUNetDenoiser
from scripts.train_diffusion import parse_diffusion_section

CFG = "config/config_diffusion.cfg"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

config = read_config(CFG)
dcfg = parse_diffusion_section(CFG)

print("loading dataset...")
ds = MeshCompletionDataset(config, 'train')
N = ds.get(0).x.shape[0]
print(f"vertices/mesh: {N}  | train size: {len(ds)}")
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
batch = next(iter(loader)).to(device)
print("batch.x", tuple(batch.x.shape), "batch.y", tuple(batch.y.shape),
      "vmask", tuple(batch.vertex_mask.shape), "B", batch.num_graphs)

print("building matrices + denoiser...")
pA, pD, pU = get_mesh_matrices(config)
print("levels (nodes):", [int(pD[0].shape[1])] + [int(d.shape[0]) for d in pD])
denoiser = GraphUNetDenoiser(list(pA), list(pD), list(pU), in_ch=7, out_ch=3,
                             base_channels=dcfg['base_channels'],
                             channel_mults=dcfg['channel_mults'],
                             time_embed_dim=dcfg['time_embed_dim']).to(device)
diffusion = GaussianDiffusion(dcfg['timesteps'], dcfg['schedule']).to(device)
print(f"denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")

print("forward + backward on one batch...")
t0 = time.time()
loss, *_ = diffusion.p_losses(denoiser, batch.y, batch.x, batch.num_graphs, N,
                              batch.vertex_mask, 1.0, 3.0)
loss.backward()
print(f"  loss={loss.item():.5f}  step time={time.time()-t0:.2f}s")
assert torch.isfinite(loss)

print("DDIM sample on one mesh (10 steps)...")
cond1 = batch.x[:N]
known_mask = cond1[:, 3] > 0.5
known_vals = cond1[:, :3].contiguous()
t0 = time.time()
with torch.no_grad():
    x0 = diffusion.ddim_sample(denoiser, cond1, 1, N, known_mask, known_vals,
                               num_steps=10, device=device)
err = (x0[known_mask] - known_vals[known_mask]).abs().max().item()
print(f"  sample shape {tuple(x0.shape)}  known max-err {err:.2e}  time={time.time()-t0:.2f}s")
assert err < 1e-5
print("\nREAL SMOKE TEST PASSED")
