"""
Graph U-Net denoiser for mesh-completion diffusion (Option B).

Operates at VERTEX resolution on a fixed-topology mesh, using the same
multi-resolution machinery as 3DGeoMeshNet (the A / D / U matrices from
utils.funcs.get_mesh_matrices):

    A[l] : edge_index (graph adjacency) at resolution level l
    D[l] : downsample matrix  level l   -> level l+1   (N_{l+1}, N_l)
    U[l] : upsample   matrix  level l+1 -> level l      (N_l, N_{l+1})

Unlike the original encoder (which mean-pools to a global latent), this is a
proper U-Net with skip connections so the decoder keeps exact vertex-level
detail — essential for diffusion denoising and for the hard known-region clamp.

Input  per node : [ noisy_offset (3) , partial_xyz (3) , mask (1) ] = 7 ch
Output per node : predicted noise eps (3)
Timestep t is injected into every block via a learned embedding (FiLM-style add).

IMPORTANT: we build a correctly *batched* edge_index per level (offsetting node
indices per graph), so convolutions are valid for batch_size > 1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import FeaStConv


# ----------------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """FeaSt graph conv residual block with timestep injection (LayerNorm + SiLU)."""

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.conv1 = FeaStConv(in_ch, out_ch)
        self.time = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.conv2 = FeaStConv(out_ch, out_ch)
        self.skip = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, h, edge_index, t_emb, B):
        n = h.shape[0] // B
        r = self.skip(h)
        x = F.silu(self.norm1(h))
        x = self.conv1(x=x, edge_index=edge_index)
        x = x + self.time(t_emb).repeat_interleave(n, dim=0)
        x = F.silu(self.norm2(x))
        x = self.conv2(x=x, edge_index=edge_index)
        return x + r


class GraphUNetDenoiser(nn.Module):
    def __init__(self, A, D, U, in_ch=7, out_ch=3,
                 base_channels=64, channel_mults=(1, 2, 4, 4), time_embed_dim=128):
        super().__init__()
        self.n_down = len(D)                      # number of down/up steps (3)
        assert len(channel_mults) == self.n_down + 1, \
            "channel_mults length must equal n_layers + 1"

        # Register A/D/U as buffers so .to(device) moves them.
        self.A = [a.long() for a in A]
        self.D = [d.float() for d in D]
        self.U = [u.float() for u in U]
        for i, a in enumerate(self.A):
            self.register_buffer(f'A_{i}', a, persistent=False)
        for i, d in enumerate(self.D):
            self.register_buffer(f'D_{i}', d, persistent=False)
        for i, u in enumerate(self.U):
            self.register_buffer(f'U_{i}', u, persistent=False)

        # Vertex counts per level: N_0 from D[0] cols, N_{l+1} from D[l] rows.
        self.num_nodes = [int(self.D[0].shape[1])] + [int(d.shape[0]) for d in self.D]

        chs = [base_channels * m for m in channel_mults]
        self.chs = chs
        self._ei_cache = {}

        # Timestep MLP.
        self.t_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input / output projections.
        self.in_proj = nn.Linear(in_ch, chs[0])
        self.out_norm = nn.LayerNorm(chs[0])
        self.out_proj = nn.Linear(chs[0], out_ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Encoder: block at level l (chs[l]) then channel-proj to chs[l+1] after downsample.
        self.enc_blocks = nn.ModuleList(
            [ResBlock(chs[l], chs[l], time_embed_dim) for l in range(self.n_down)]
        )
        self.down_proj = nn.ModuleList(
            [nn.Linear(chs[l], chs[l + 1]) for l in range(self.n_down)]
        )

        # Bottleneck.
        self.mid_block = ResBlock(chs[self.n_down], chs[self.n_down], time_embed_dim)

        # Decoder: upsample, channel-proj chs[l+1]->chs[l], concat skip, block.
        self.up_proj = nn.ModuleList(
            [nn.Linear(chs[l + 1], chs[l]) for l in range(self.n_down)]
        )
        self.dec_blocks = nn.ModuleList(
            [ResBlock(2 * chs[l], chs[l], time_embed_dim) for l in range(self.n_down)]
        )

    # -- batched edge_index (offset node indices per graph) -----------------
    def _batched_edge_index(self, level, B, device):
        key = (level, B)
        cached = self._ei_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        base = getattr(self, f'A_{level}').to(device)          # (2, E)
        N = self.num_nodes[level]
        E = base.shape[1]
        offs = (torch.arange(B, device=device) * N).repeat_interleave(E)
        ei = base.repeat(1, B) + offs.unsqueeze(0)
        self._ei_cache[key] = ei
        return ei

    def _downsample(self, h, level, B):
        D = getattr(self, f'D_{level}')
        N_in = self.num_nodes[level]
        C = h.shape[1]
        h = h.view(B, N_in, C)
        h = torch.einsum('ij,bjc->bic', D, h)                  # (B, N_out, C)
        return h.reshape(-1, C)

    def _upsample(self, h, level, B):
        U = getattr(self, f'U_{level}')
        N_in = self.num_nodes[level + 1]
        C = h.shape[1]
        h = h.view(B, N_in, C)
        h = torch.einsum('ij,bjc->bic', U, h)                  # (B, N_out, C)
        return h.reshape(-1, C)

    # -- forward ------------------------------------------------------------
    def forward(self, delta_t, cond, t, B):
        """
        delta_t : (B*N0, 3) noisy offset field
        cond    : (B*N0, 4) [partial_xyz, mask]
        t       : (B,)      integer timesteps
        B       : batch size (num graphs)
        returns : (B*N0, 3) predicted noise eps
        """
        device = delta_t.device
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))

        h = self.in_proj(torch.cat([delta_t, cond], dim=-1))

        skips = []
        for l in range(self.n_down):
            ei = self._batched_edge_index(l, B, device)
            h = self.enc_blocks[l](h, ei, t_emb, B)
            skips.append(h)
            h = self._downsample(h, l, B)
            h = self.down_proj[l](h)

        ei_mid = self._batched_edge_index(self.n_down, B, device)
        h = self.mid_block(h, ei_mid, t_emb, B)

        for l in reversed(range(self.n_down)):
            h = self._upsample(h, l, B)
            h = self.up_proj[l](h)
            h = torch.cat([h, skips[l]], dim=-1)
            ei = self._batched_edge_index(l, B, device)
            h = self.dec_blocks[l](h, ei, t_emb, B)

        h = F.silu(self.out_norm(h))
        return self.out_proj(h)
