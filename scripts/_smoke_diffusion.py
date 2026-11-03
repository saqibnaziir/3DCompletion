"""Smoke test for the diffusion plumbing — tiny synthetic mesh, CPU, no data needed."""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.diffusion import GaussianDiffusion, EMA
from utils.diffusion_model import GraphUNetDenoiser


def make_fake_matrices():
    # 4 levels: N0=8, N1=4, N2=2, N3=1 (down factor 2 each)
    Ns = [8, 4, 2, 1]

    def ring_edges(n):
        if n == 1:
            return torch.zeros(2, 0, dtype=torch.long)
        src = torch.arange(n)
        dst = (src + 1) % n
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        return ei.long()

    A = [ring_edges(n) for n in Ns]
    D, U = [], []
    for l in range(3):
        n_in, n_out = Ns[l], Ns[l + 1]
        d = torch.zeros(n_out, n_in)
        for i in range(n_out):
            d[i, 2 * i] = 0.5
            d[i, 2 * i + 1] = 0.5
        D.append(d)
        U.append(d.t().contiguous() * 2.0)  # crude upsample
    return A, D, U


def main():
    torch.manual_seed(0)
    A, D, U = make_fake_matrices()
    N = 8
    B = 3

    denoiser = GraphUNetDenoiser(A, D, U, in_ch=7, out_ch=3,
                                 base_channels=16, channel_mults=(1, 2, 4, 4),
                                 time_embed_dim=32)
    diffusion = GaussianDiffusion(timesteps=50, schedule='cosine')
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"denoiser params: {n_params:,}")

    # Fake batch
    x0 = torch.randn(B * N, 3)
    cond = torch.cat([torch.randn(B * N, 3), torch.randint(0, 2, (B * N, 1)).float()], dim=1)
    vmask = cond[:, 3] < 0.5  # missing where mask==0

    # 1) forward
    t = torch.randint(0, 50, (B,))
    eps = denoiser(x0, cond, t, B)
    assert eps.shape == (B * N, 3), eps.shape
    print("forward OK:", eps.shape)

    # 2) training loss
    loss, eps_pred, t2, x_t, noise = diffusion.p_losses(
        denoiser, x0, cond, B, N, vmask, lambda_known=1.0, lambda_missing=3.0)
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in denoiser.parameters() if p.grad is not None)
    print(f"loss OK: {loss.item():.4f}  grad-sum {gnorm:.2f}")
    assert torch.isfinite(loss)

    # 3) EMA update
    ema = EMA(denoiser, 0.99); ema.update(denoiser)
    print("EMA OK")

    # 4) DDIM sampling with hard clamp (B=1)
    cond1 = cond[:N]
    known_mask = cond1[:, 3] > 0.5
    known_vals = cond1[:, :3].contiguous()
    x0s = diffusion.ddim_sample(denoiser, cond1, B=1, nodes_per_graph=N,
                                known_mask=known_mask, known_values=known_vals,
                                num_steps=10, device='cpu')
    assert x0s.shape == (N, 3)
    # known region must be exact
    err = (x0s[known_mask] - known_vals[known_mask]).abs().max().item()
    print(f"DDIM OK: {x0s.shape}  known-region max-err {err:.2e} (should be ~0)")
    assert err < 1e-5, "known-region clamp failed"

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
