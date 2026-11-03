"""
Gaussian diffusion utilities for 3D mesh completion (Option B).

We diffuse the per-vertex DEFORMATION (offset) field of a fixed-topology mesh:

    V_complete = T + Delta          (T = mean template)

Because normalization subtracts the mean and the template *is* the mean, the
offset field Delta equals the normalized vertex positions. So the diffusion
target x0 is simply the normalized complete mesh (== dataset `data.y`), and the
conditioning is the partial observation (== dataset `data.x` = [partial_xyz, mask]).

Key design choices:
  - We diffuse in *vertex space* (not latent space) so the known region can be
    clamped EXACTLY at every sampling step (hard data-consistency, RePaint-style).
  - Configurable prediction target: eps (noise), x0 (clean position), or v
    (velocity) — set via `predict`. Cosine schedule (Nichol & Dhariwal).
  - DDIM sampling (eta=0) for fast, deterministic-per-noise completion.

The denoiser network itself lives in utils/diffusion_model.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Noise schedules
# ----------------------------------------------------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


# ----------------------------------------------------------------------------
# Gaussian diffusion
# ----------------------------------------------------------------------------
class GaussianDiffusion(nn.Module):
    """
    Holds the diffusion schedule as buffers and provides q_sample (forward
    noising), the training loss, and DDIM sampling with hard known-region
    clamping. Has NO trainable parameters; the denoiser is passed in.

    Node convention: all tensors are flat (M, C) where M = B * N (B graphs of N
    vertices, fixed topology). Timesteps `t` are per-graph (B,) and expanded to
    per-node internally via `nodes_per_graph`.
    """

    def __init__(self, timesteps=1000, schedule='cosine', predict='eps'):
        super().__init__()
        self.timesteps = timesteps
        if predict not in ('eps', 'x0', 'v'):
            raise ValueError(f"Unknown prediction target: {predict}")
        self.predict = predict
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))

    # -- helpers ------------------------------------------------------------
    def _expand(self, coeff_1d, t_per_node):
        """Gather schedule coefficient at integer timesteps, shape -> (M, 1)."""
        return coeff_1d[t_per_node].unsqueeze(-1)

    def q_sample(self, x0, t_per_node, noise):
        """Forward diffusion: x_t = sqrt(acp) x0 + sqrt(1-acp) noise."""
        s_acp = self._expand(self.sqrt_alphas_cumprod, t_per_node)
        s_omacp = self._expand(self.sqrt_one_minus_alphas_cumprod, t_per_node)
        return s_acp * x0 + s_omacp * noise

    def predict_x0_from_eps(self, x_t, t_per_node, eps):
        s_acp = self._expand(self.sqrt_alphas_cumprod, t_per_node)
        s_omacp = self._expand(self.sqrt_one_minus_alphas_cumprod, t_per_node)
        return (x_t - s_omacp * eps) / s_acp

    def get_v(self, x0, noise, t_per_node):
        """Velocity target v = sqrt(acp) noise - sqrt(1-acp) x0 (v-prediction)."""
        s_acp = self._expand(self.sqrt_alphas_cumprod, t_per_node)
        s_omacp = self._expand(self.sqrt_one_minus_alphas_cumprod, t_per_node)
        return s_acp * noise - s_omacp * x0

    def target_for(self, x0, noise, t_per_node):
        """The regression target the network is trained to output for `self.predict`."""
        if self.predict == 'eps':
            return noise
        if self.predict == 'x0':
            return x0
        return self.get_v(x0, noise, t_per_node)   # 'v'

    def decode(self, model_out, x_t, t_per_node):
        """Map a raw network output to (x0, eps) according to `self.predict`.

        x0-pred/v-pred recover x0 *directly* (no division by sqrt(acp)), so they
        do not blow up at high noise and need no clipping — unlike eps-pred.
        """
        s_acp = self._expand(self.sqrt_alphas_cumprod, t_per_node)
        s_omacp = self._expand(self.sqrt_one_minus_alphas_cumprod, t_per_node)
        if self.predict == 'eps':
            eps = model_out
            x0 = (x_t - s_omacp * eps) / s_acp
        elif self.predict == 'x0':
            x0 = model_out
            eps = (x_t - s_acp * x0) / s_omacp
        else:  # 'v': x0 = sqrt(acp) x_t - sqrt(1-acp) v ; eps = sqrt(1-acp) x_t + sqrt(acp) v
            v = model_out
            x0 = s_acp * x_t - s_omacp * v
            eps = s_omacp * x_t + s_acp * v
        return x0, eps

    # -- training loss ------------------------------------------------------
    def p_losses(self, denoiser, x0, cond, B, nodes_per_graph,
                 vertex_mask, lambda_known=1.0, lambda_missing=3.0,
                 vertex_weight=None):
        """
        Single denoising-score-matching step.

        Args:
            denoiser: callable(delta_t, cond, t_per_graph, B) -> model_out (M,3)
            x0:       (M,3) clean offset field (== normalized complete verts)
            cond:     (M,4) conditioning [partial_xyz, mask_channel]
            B:        number of graphs in the batch
            nodes_per_graph: N (vertices per mesh, fixed)
            vertex_mask: (M,) bool, True = missing vertex
            lambda_known/lambda_missing: per-region loss weights (flat)
            vertex_weight: (M,) optional per-vertex weight that REPLACES the
                flat two-region split -- e.g. the geodesic depth ramp
                w = 1 + (w_max-1)*d/d_max from utils.boundary_field.

        Returns:
            loss (scalar), model_out (M,3), t_per_graph (B,), x_t (M,3), noise (M,3)
        """
        device = x0.device
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)
        t_per_node = t.repeat_interleave(nodes_per_graph)

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t_per_node, noise)

        model_out = denoiser(x_t, cond, t, B)
        target = self.target_for(x0, noise, t_per_node)

        # Per-region weighting of the prediction-MSE. Under x0-prediction this
        # weights *positional* error directly (the missing region we care about);
        # under eps-prediction it only weights noise-prediction error.
        if vertex_weight is not None:
            weight = vertex_weight.to(dtype=x0.dtype)
        else:
            weight = torch.where(vertex_mask,
                                 torch.full_like(vertex_mask, lambda_missing, dtype=x0.dtype),
                                 torch.full_like(vertex_mask, lambda_known, dtype=x0.dtype))
        loss = (weight.unsqueeze(-1) * (model_out - target) ** 2).mean()
        return loss, model_out, t, x_t, noise

    # -- DDIM sampling with hard data-consistency ---------------------------
    @torch.no_grad()
    def ddim_sample(self, denoiser, cond, B, nodes_per_graph, known_mask,
                    known_values, num_steps=50, device='cpu', eta=0.0,
                    progress=False, clip_value=None):
        """
        Completion sampling. Generates the offset field for the MISSING region
        while clamping the KNOWN region to its exact value at every step
        (RePaint-style hard data-consistency in vertex space).

        Args:
            denoiser: callable(delta_t, cond, t_per_graph, B) -> eps_pred
            cond:        (M,4) [partial_xyz, mask]
            B:           num graphs (usually 1 at inference, or K for diversity)
            nodes_per_graph: N
            known_mask:  (M,) bool, True = known vertex (clamp these)
            known_values:(M,3) exact offsets for known vertices (== partial_xyz)
            num_steps:   DDIM steps
        Returns:
            x0:          (M,3) sampled offset field (known region exact)
        """
        M = cond.shape[0]
        x = torch.randn(M, 3, device=device)

        # Even DDIM subsequence of timesteps, high -> low.
        step_seq = torch.linspace(self.timesteps - 1, 0, num_steps,
                                  device=device).round().long()

        iterator = range(num_steps)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DDIM", leave=False)

        for i in iterator:
            t = step_seq[i]
            t_per_node = t.repeat(M)
            t_per_graph = t.repeat(B)

            model_out = denoiser(x, cond, t_per_graph, B)
            x0, eps = self.decode(model_out, x, t_per_node)

            # Stabilize: clamp the predicted x0 (standardized space). Only needed
            # for eps-prediction, where x0 = (x_t - sqrt(1-acp) eps) / sqrt(acp)
            # divides by ~0 and explodes at high noise. x0/v-prediction output x0
            # directly (bounded), so no clip is required there.
            if clip_value is not None and self.predict == 'eps':
                x0 = x0.clamp(-clip_value, clip_value)

            # Hard data-consistency: overwrite known region with exact target.
            x0 = torch.where(known_mask.unsqueeze(-1), known_values, x0)

            if i < num_steps - 1:
                t_next = step_seq[i + 1]
                acp_next = self.alphas_cumprod[t_next]
                s_acp_next = torch.sqrt(acp_next)
                s_omacp_next = torch.sqrt(1.0 - acp_next)

                # DDIM (eta=0): deterministic direction from predicted eps.
                x = s_acp_next * x0 + s_omacp_next * eps

                # Re-noise the KNOWN region from its exact value (RePaint).
                known_noise = torch.randn_like(x)
                x_known = s_acp_next * known_values + s_omacp_next * known_noise
                x = torch.where(known_mask.unsqueeze(-1), x_known, x)
            else:
                x = x0

        return x


# ----------------------------------------------------------------------------
# Exponential Moving Average of weights (standard for diffusion quality)
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
