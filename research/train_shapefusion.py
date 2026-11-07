"""
ShapeFusion-style baseline (masked-diffusion on registered meshes) — the
"collision paper" comparison, reimplemented faithfully on OUR backbone / data /
metric so it is directly scorable by research/eval_completion_dir.py.

WHAT THIS REPRODUCES (ShapeFusion, ECCV 2024, arXiv 2403.19773):
  ShapeFusion diffuses registered face/body meshes with a *masked-diffusion*
  scheme: the un-edited (known) region is held at its clean value inside the
  diffused tensor and the model inpaints the rest, so preservation is *implicit*
  through the mask rather than an explicit per-step RePaint projection.

HOW IT DIFFERS FROM OUR MeshOffsetDiff (the two things this baseline isolates):
  1. TRAINING.  Ours noises the WHOLE offset field x_t = q_sample(x0) and lets the
     net reconstruct the known region from noise, with the observation supplied
     only through the separate cond channel. ShapeFusion-style instead keeps the
     KNOWN region CLEAN inside x_t (x_t[known] = x0[known]) and computes the loss
     on the MISSING region only — a conditional inpainter.
  2. SAMPLING.  Ours uses RePaint hard data-consistency (overwrite + RE-NOISE the
     known region each step). ShapeFusion-style keeps the known region CLEAN at
     every step (implicit preservation), matching how it was trained.

SCOPING NOTE (honest, for the paper): in this registered COMA setting the
"normalized vertices" and the "offset field" coincide (template == per-vertex
mean), so ShapeFusion's "raw-vertex vs residual-offset" axis is not separable
here; both models therefore share the same unit-variance standardization (without
it EITHER diverges — that is ablation (a), not a ShapeFusion difference). We also
keep our graph-U-Net backbone rather than ShapeFusion's vertex-index positional
encoding. What is compared is thus the masked-diffusion training + implicit
preservation scheme, on identical data, topology and metric.

Writes to its OWN checkpoint dir (default data/checkpoints_shapefusion) so it
never clobbers the main diffusion model.

RESUME-SAFE: a full state (model+ema+optimizer+scheduler+epoch+best) is written
every epoch to shapefusion_state.pt, and a per-epoch line to train_log.csv, so a
reboot mid-run costs at most one epoch — just relaunch with --resume.

Usage (conda 3dmesh, GPU):
  # train (fresh)
  python research/train_shapefusion.py train \
      --config_file research/configs/config_diffusion.cfg --epochs 300

  # resume after a restart (picks up from the last completed epoch)
  python research/train_shapefusion.py train \
      --config_file research/configs/config_diffusion.cfg --epochs 300 --resume

  # complete the test set -> completed_shapefusion/*.ply
  python research/train_shapefusion.py eval \
      --config_file research/configs/config_diffusion.cfg

  # score with the shared harness (appends a row to baseline_results.csv)
  python research/eval_completion_dir.py \
      --completed_dir data/completion_test_samples/test_set/completed_shapefusion \
      --suffix _completed_shapefusion.ply --method ShapeFusion-style \
      --out_csv data/completion_test_samples/baseline_results.csv
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import random
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from trimesh import Trimesh, load_mesh
from torch_geometric.loader import DataLoader

from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion, EMA
from utils.diffusion_model import GraphUNetDenoiser
from scripts.train_diffusion import parse_diffusion_section
from scripts.infer_diffusion import load_norm

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEFAULT_CKPT = os.path.join('data', 'checkpoints_shapefusion')


# ---------------------------------------------------------------------------
def build_net(config, dcfg, device):
    pA, pD, pU = get_mesh_matrices(config)
    return GraphUNetDenoiser(
        list(pA), list(pD), list(pU),
        in_ch=7, out_ch=3,
        base_channels=dcfg['base_channels'],
        channel_mults=dcfg['channel_mults'],
        time_embed_dim=dcfg['time_embed_dim'],
    ).to(device)


def masked_p_losses(diffusion, denoiser, x0, cond, B, N, vmask):
    """
    ShapeFusion-style masked-diffusion loss: keep the KNOWN region clean inside
    x_t, noise only the MISSING region, and score eps on the missing region only.
    """
    device = x0.device
    t = torch.randint(0, diffusion.timesteps, (B,), device=device, dtype=torch.long)
    t_per_node = t.repeat_interleave(N)

    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t_per_node, noise)
    # implicit preservation: known region stays at its clean value in x_t
    known = ~vmask
    x_t = torch.where(known.unsqueeze(-1), x0, x_t)

    eps_pred = denoiser(x_t, cond, t, B)

    if vmask.any():
        loss = ((eps_pred[vmask] - noise[vmask]) ** 2).mean()
    else:                                    # degenerate (no missing) — skip
        loss = ((eps_pred - noise) ** 2).mean() * 0.0
    return loss


@torch.no_grad()
def masked_ddim_sample(diffusion, denoiser, cond, known_mask, known_values,
                       N, device, num_steps=50, clip_value=None):
    """
    DDIM sampling that keeps the KNOWN region CLEAN at every step (implicit
    preservation, no RePaint re-noise) — matches masked-diffusion training.
    Single graph (B=1). Returns x0 (N,3) in standardized space, known == exact.
    """
    M = cond.shape[0]
    x = torch.randn(M, 3, device=device)
    x = torch.where(known_mask.unsqueeze(-1), known_values, x)      # clean known
    step_seq = torch.linspace(diffusion.timesteps - 1, 0, num_steps,
                              device=device).round().long()

    for i in range(num_steps):
        t = step_seq[i]
        t_per_node = t.repeat(M)
        eps = denoiser(x, cond, t.repeat(1), 1)
        x0 = diffusion.predict_x0_from_eps(x, t_per_node, eps)
        if clip_value is not None:
            x0 = x0.clamp(-clip_value, clip_value)
        x0 = torch.where(known_mask.unsqueeze(-1), known_values, x0)

        if i < num_steps - 1:
            t_next = step_seq[i + 1]
            acp_next = diffusion.alphas_cumprod[t_next]
            x = torch.sqrt(acp_next) * x0 + torch.sqrt(1.0 - acp_next) * eps
            # keep known region CLEAN (implicit preservation, no re-noise)
            x = torch.where(known_mask.unsqueeze(-1), known_values, x)
        else:
            x = x0
    return x


# ---------------------------------------------------------------------------
def cmd_train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    if args.epochs is not None:
        config['epoch'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, 'shapefusion_state.pt')
    log_path = os.path.join(ckpt_dir, 'train_log.csv')

    ds_train = MeshCompletionDataset(config, 'train')
    ds_eval = MeshCompletionDataset(config, 'eval')
    N = ds_train.get(0).x.shape[0]
    print(f"Vertices/mesh: {N} | train={len(ds_train)} eval={len(ds_eval)}")

    sigma = ds_train.data.y.std(dim=0).clamp(min=1e-6).to(device)
    clip_value = dcfg.get('sampling_clip', 6.0)
    torch.save({'sigma': sigma.cpu(), 'clip_value': clip_value},
               os.path.join(ckpt_dir, 'shapefusion_scale.pt'))
    print(f"Offset sigma (per-axis std): {sigma.tolist()} | clip={clip_value}")

    nw = config['num_workers']
    tl = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=nw > 0)
    el = DataLoader(ds_eval, batch_size=config['batch_size'], shuffle=False,
                    num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=nw > 0)

    denoiser = build_net(config, dcfg, device)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule']).to(device)
    ema = EMA(denoiser, decay=dcfg['ema_decay'])
    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"Denoiser parameters: {n_params:,} (same backbone as MeshOffsetDiff)")

    opt = torch.optim.Adam(denoiser.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['epoch'], eta_min=config['lr'] * 0.05)

    start_epoch, best = 1, float('inf')
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        denoiser.load_state_dict(st['model'])
        ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['optimizer'])
        sched.load_state_dict(st['scheduler'])
        start_epoch = st['epoch'] + 1
        best = st['best_eval']
        print(f"** resumed from epoch {st['epoch']} (best eval={best:.6f}) "
              f"-> continuing at epoch {start_epoch} **")
    elif args.resume:
        print(f"--resume given but no {state_path}; starting fresh.")

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train', 'eval', 'lr', 'timestamp'])

    def run(loader, train):
        denoiser.train(train)
        tot, seen = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc='train' if train else 'eval', leave=False):
                batch = batch.to(device)
                B = batch.num_graphs
                x0 = batch.y / sigma
                cond = batch.x.clone()
                cond[:, :3] = cond[:, :3] / sigma
                vmask = batch.vertex_mask
                loss = masked_p_losses(diffusion, denoiser, x0, cond, B, N, vmask)
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                    opt.step(); ema.update(denoiser)
                tot += B * loss.item(); seen += B
        return tot / max(seen, 1)

    print(f"Starting ShapeFusion-style training (epochs {start_epoch}..{config['epoch']})")
    for ep in range(start_epoch, config['epoch'] + 1):
        tr = run(tl, True)
        ev = run(el, False)
        lr_now = sched.get_last_lr()[0]
        print(f"Epoch {ep:4d} | train {tr:.6f} | eval {ev:.6f} | lr {lr_now:.2e}")
        sched.step()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f'{tr:.6f}', f'{ev:.6f}', f'{lr_now:.3e}',
                                    datetime.now().isoformat(timespec='seconds')])

        if ev < best:
            best = ev
            torch.save(denoiser.state_dict(),
                       os.path.join(ckpt_dir, 'shapefusion_denoiser.pt'))
            torch.save(ema.state_dict(),
                       os.path.join(ckpt_dir, 'shapefusion_denoiser_ema.pt'))
            print(f"  ** saved best (eval={ev:.6f}) **")

        # resumable full state, overwritten every epoch
        torch.save({'epoch': ep, 'best_eval': best,
                    'model': denoiser.state_dict(), 'ema': ema.state_dict(),
                    'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                    'sigma': sigma.cpu(), 'clip_value': clip_value}, state_path)

    print("Training complete.")


# ---------------------------------------------------------------------------
@torch.no_grad()
def cmd_eval(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = read_config(args.config_file)
    dcfg = parse_diffusion_section(args.config_file)
    ckpt_dir = args.ckpt_dir

    denoiser = build_net(config, dcfg, device)
    ema_p = os.path.join(ckpt_dir, 'shapefusion_denoiser_ema.pt')
    raw_p = os.path.join(ckpt_dir, 'shapefusion_denoiser.pt')
    path = ema_p if (not args.no_ema and os.path.exists(ema_p)) else raw_p
    print(f"Loading weights: {path}")
    denoiser.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    denoiser.eval()
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule']).to(device)

    scale = torch.load(os.path.join(ckpt_dir, 'shapefusion_scale.pt'),
                       map_location=device, weights_only=False)
    sigma = scale['sigma'].to(device)
    sigma_np = sigma.cpu().numpy().astype(np.float32)
    clip_value = scale.get('clip_value', None)
    num_steps = args.steps if args.steps is not None else dcfg['sampling_steps']

    mean_np, std_np, mean_only = load_norm(config)
    mean_np = mean_np.numpy() if hasattr(mean_np, 'numpy') else np.asarray(mean_np)
    std_np = (std_np.numpy() if (std_np is not None and hasattr(std_np, 'numpy'))
              else std_np)

    template = load_mesh(config['template_file'], process=False)
    faces = np.array(template.faces)
    N_t = len(template.vertices)

    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples', 'test_set')
    out_dir = os.path.join(test_dir, 'completed_shapefusion')
    os.makedirs(out_dir, exist_ok=True)
    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    print(f"Completing {len(gt_files)} test samples -> {out_dir} "
          f"(DDIM steps={num_steps})\n")

    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(test_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        mask = np.load(mask_file).astype(bool)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t or len(mask) != N_t or not mask.any():
            print(f"  skip {base}: size/mask mismatch"); continue

        partial = gt.copy(); partial[mask] = 0.0
        vn = partial.astype(np.float32) - mean_np
        if not mean_only and std_np is not None:
            vn = vn / std_np
        vn[mask] = 0.0
        cond_xyz = vn / sigma_np; cond_xyz[mask] = 0.0
        mch = np.ones((N_t, 1), np.float32); mch[mask] = 0.0
        cond = torch.tensor(np.concatenate([cond_xyz, mch], 1), device=device)
        known_mask = cond[:, 3] > 0.5
        known_values = cond[:, :3].contiguous()

        x0 = masked_ddim_sample(diffusion, denoiser, cond, known_mask, known_values,
                                N_t, device, num_steps=num_steps, clip_value=clip_value)
        out = x0.cpu().numpy() * sigma_np
        if not mean_only and std_np is not None:
            out = out * std_np
        out = out + mean_np

        comp = gt.copy(); comp[mask] = out[mask]          # blend (known exact)
        Trimesh(vertices=comp, faces=faces, process=False).export(
            os.path.join(out_dir, f'{base}_completed_shapefusion.ply'))
        print(f"  {base}: missing {int(mask.sum())} done")

    print(f"\nDone. Score with research/eval_completion_dir.py on {out_dir}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='ShapeFusion-style masked-diffusion baseline')
    sub = ap.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--config_file', required=True)
    pt.add_argument('--epochs', type=int, default=None)
    pt.add_argument('--batch_size', type=int, default=None)
    pt.add_argument('--ckpt_dir', default=DEFAULT_CKPT)
    pt.add_argument('--resume', action='store_true',
                    help='continue from shapefusion_state.pt if present')
    pt.add_argument('--seed', type=int, default=1)

    pe = sub.add_parser('eval')
    pe.add_argument('--config_file', required=True)
    pe.add_argument('--ckpt_dir', default=DEFAULT_CKPT)
    pe.add_argument('--test_dir', default=None)
    pe.add_argument('--steps', type=int, default=None)
    pe.add_argument('--no_ema', action='store_true')

    args = ap.parse_args()
    seed = getattr(args, 'seed', 1)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if args.cmd == 'train':
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == '__main__':
    main()
