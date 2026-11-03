"""
Measure the position vs normal loss-term balance over MANY batches, using the
actual converged depthw weights, so lambda_normal is set from a stable estimate
rather than one noisy batch.

Both terms depend on the sampled timestep t (one t per graph), so a single batch
is a poor estimate of either. This averages over N batches and reports the
lambda that puts the weighted normal term at 0.1x / 0.2x / 0.3x of the position
term.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from trimesh import load_mesh
from torch_geometric.loader import DataLoader

from config.config import read_config, resolve_config_file
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion
from utils.diffusion_model import GraphUNetDenoiser
from utils.normal_loss import batched_faces, normal_cosine_loss
from scripts.train_diffusion import parse_diffusion_section

N_BATCHES = 25
CKPT = 'predict_x0_depthw'


def main():
    cfg_path = resolve_config_file('COMA', None)
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)
    dcfg['predict'] = 'x0'
    # Match the fine-tune's loss weighting exactly.
    config['depth_weight'] = True
    config['depth_weight_normalize'] = True
    config.setdefault('depth_weight_max', config.get('lambda_missing', 3.0))
    device = torch.device('cuda:0')

    ds = MeshCompletionDataset(config, 'train')
    N = ds.get(0).x.shape[0]
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True,
                        num_workers=0, drop_last=True)

    ckpt_dir = os.path.join(config['checkpoint_dir'], CKPT)
    scale = torch.load(os.path.join(ckpt_dir, 'diffusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma = scale['sigma'].to(device)

    pA, pD, pU = get_mesh_matrices(config)
    denoiser = GraphUNetDenoiser(list(pA), list(pD), list(pU), in_ch=7, out_ch=3,
                                 base_channels=dcfg['base_channels'],
                                 channel_mults=dcfg['channel_mults'],
                                 time_embed_dim=dcfg['time_embed_dim']).to(device)
    # Training continues from the RAW weights (not EMA), so calibrate on those.
    denoiser.load_state_dict(torch.load(os.path.join(ckpt_dir, 'diffusion_denoiser.pt'),
                                        map_location=device, weights_only=False))
    denoiser.eval()
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'], predict='x0').to(device)

    tpl = load_mesh(config['template_file'], process=False)
    faces = torch.as_tensor(np.asarray(tpl.faces), dtype=torch.long, device=device)

    torch.manual_seed(0)
    pos_terms, nrm_terms, ratios = [], [], []
    fcache = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            batch = batch.to(device)
            B = batch.num_graphs
            x0 = batch.y / sigma
            cond = batch.x.clone()
            cond[:, :3] = cond[:, :3] / sigma
            vmask = batch.vertex_mask
            vw = getattr(batch, 'loss_weight', None)

            loss, model_out, t, x_t, _ = diffusion.p_losses(
                denoiser, x0, cond, B, N, vmask,
                lambda_known=1.0, lambda_missing=3.0, vertex_weight=vw)

            t_node = t.repeat_interleave(N)
            x0_pred, _ = diffusion.decode(model_out, x_t, t_node)
            if B not in fcache:
                fcache[B] = batched_faces(faces, B, N, device)
            n_loss = normal_cosine_loss(x0_pred, x0, fcache[B], sigma,
                                        region_mask=vmask,
                                        snr_weight=diffusion.alphas_cumprod[t_node])
            pos_terms.append(float(loss))
            nrm_terms.append(float(n_loss))
            ratios.append(float(n_loss) / max(float(loss), 1e-12))

    p, n = float(np.mean(pos_terms)), float(np.mean(nrm_terms))
    print(f"\nover {len(pos_terms)} batches (batch_size={config['batch_size']}, "
          f"weights={CKPT}/diffusion_denoiser.pt)")
    print(f"  position term : {p:.6f}  (per-batch sd {np.std(pos_terms):.6f}, "
          f"min {min(pos_terms):.6f} max {max(pos_terms):.6f})")
    print(f"  normal   term : {n:.6f}  (per-batch sd {np.std(nrm_terms):.6f}, "
          f"min {min(nrm_terms):.6f} max {max(nrm_terms):.6f})")
    print(f"  raw ratio n/p : {n/p:.2f}  (per-batch sd {np.std(ratios):.2f})")
    print(f"\nlambda_normal for a target weighted share of the position term:")
    for tgt in (0.1, 0.2, 0.3):
        print(f"  {tgt:.1f}x  ->  lambda_normal = {tgt * p / n:.5f}")


if __name__ == '__main__':
    main()
