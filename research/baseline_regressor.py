"""
Deterministic-regressor baseline (ablation: SAME backbone, no diffusion).

This is the single most important comparison for the paper: it isolates what the
diffusion process contributes by holding the architecture fixed. We reuse the
exact `GraphUNetDenoiser` graph U-Net, but instead of denoising we feed a zero
noisy-offset channel and timestep t=0 and train the network to regress the
standardized offset field directly (region-weighted MSE). Inference is a single
forward pass; known vertices are blended back exactly (same protocol as our
diffusion model), so known-region error is 0.

If this regressor matches the diffusion model on mean error, the diffusion's
value is "only" diversity + uncertainty; if diffusion wins, the generative
process is contributing accuracy too. Either outcome is a publishable result.

Usage (same config as the diffusion model, so dataset/arch/sigma are identical):
  # train
  python research/baseline_regressor.py train \
      --config_file research/configs/config_diffusion.cfg --epochs 300

  # complete the test set -> writes completed_regressor/*.ply
  python research/baseline_regressor.py eval \
      --config_file research/configs/config_diffusion.cfg

  # then score with the shared harness:
  python research/eval_completion_dir.py \
      --completed_dir data/completion_test_samples/test_set/completed_regressor \
      --suffix _completed_regressor.ply --method Regressor-sameNet \
      --out_csv data/completion_test_samples/baseline_results.csv
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import configparser
import random
import warnings

import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from trimesh import Trimesh, load_mesh
from torch_geometric.loader import DataLoader

from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import EMA
from utils.diffusion_model import GraphUNetDenoiser

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEFAULT_CKPT = os.path.join('data', 'checkpoints_regressor')


def parse_arch(config_file):
    cp = configparser.RawConfigParser()
    cp.read(config_file)
    sec = 'diffusion parameters'
    return dict(
        base_channels=cp.getint(sec, 'base_channels', fallback=64),
        channel_mults=tuple(int(x) for x in
                            cp.get(sec, 'channel_mults', fallback='1,2,4,4').split(',')),
        time_embed_dim=cp.getint(sec, 'time_embed_dim', fallback=128),
    )


def build_net(config, arch, device):
    pA, pD, pU = get_mesh_matrices(config)
    net = GraphUNetDenoiser(
        list(pA), list(pD), list(pU),
        in_ch=7, out_ch=3,
        base_channels=arch['base_channels'],
        channel_mults=arch['channel_mults'],
        time_embed_dim=arch['time_embed_dim'],
    ).to(device)
    return net


def region_weighted_mse(pred, target, vmask, lam_known, lam_missing):
    w = torch.where(vmask, torch.as_tensor(lam_missing, device=pred.device),
                    torch.as_tensor(lam_known, device=pred.device)).float()
    se = ((pred - target) ** 2).sum(dim=-1)            # (B*N,)
    return (w * se).sum() / w.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
def cmd_train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = read_config(args.config_file)
    arch = parse_arch(args.config_file)
    if args.epochs is not None:
        config['epoch'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    ds_train = MeshCompletionDataset(config, 'train')
    ds_eval = MeshCompletionDataset(config, 'eval')
    N = ds_train.get(0).x.shape[0]
    print(f"Vertices/mesh: {N} | train={len(ds_train)} eval={len(ds_eval)}")

    sigma = ds_train.data.y.std(dim=0).clamp(min=1e-6).to(device)
    torch.save({'sigma': sigma.cpu()}, os.path.join(ckpt_dir, 'regressor_scale.pt'))
    print(f"Offset sigma (per-axis std): {sigma.tolist()}")

    nw = config['num_workers']
    tl = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=nw > 0)
    el = DataLoader(ds_eval, batch_size=config['batch_size'], shuffle=False,
                    num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=nw > 0)

    net = build_net(config, arch, device)
    ema = EMA(net, decay=0.9995)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Regressor parameters: {n_params:,} (same backbone as diffusion)")

    opt = torch.optim.Adam(net.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['epoch'], eta_min=config['lr'] * 0.05)

    lam_known = config.get('lambda_known', 1.0)
    lam_missing = config.get('lambda_missing', 3.0)

    def run(loader, train):
        net.train(train)
        tot, seen = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc='train' if train else 'eval', leave=False):
                batch = batch.to(device)
                B = batch.num_graphs
                x0 = batch.y / sigma                       # (B*N,3) standardized target
                cond = batch.x.clone()
                cond[:, :3] = cond[:, :3] / sigma          # (B*N,4)
                vmask = batch.vertex_mask
                zeros = torch.zeros_like(x0)               # no noisy channel
                t0 = torch.zeros(B, device=device, dtype=torch.long)
                pred = net(zeros, cond, t0, B)             # predict standardized offset
                loss = region_weighted_mse(pred, x0, vmask, lam_known, lam_missing)
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step(); ema.update(net)
                tot += B * loss.item(); seen += B
        return tot / max(seen, 1)

    best = float('inf')
    for ep in range(1, config['epoch'] + 1):
        tr = run(tl, True)
        ev = run(el, False)
        print(f"Epoch {ep:4d} | train {tr:.6f} | eval {ev:.6f} | lr {sched.get_last_lr()[0]:.2e}")
        sched.step()
        if ev < best:
            best = ev
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'regressor.pt'))
            torch.save(ema.state_dict(), os.path.join(ckpt_dir, 'regressor_ema.pt'))
            print(f"  ** saved best (eval={ev:.6f}) **")
    print("Training complete.")


# ---------------------------------------------------------------------------
def load_norm(config):
    p = os.path.join(config['dataset_dir'], 'norm_completion.pt')
    if not os.path.exists(p):
        p = os.path.join(config['dataset_dir'], 'norm.pt')
    nd = torch.load(p, map_location='cpu', weights_only=False)
    return nd['mean'].numpy(), (nd['std'].numpy() if nd['std'] is not None else None), nd['std'] is None


@torch.no_grad()
def cmd_eval(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = read_config(args.config_file)
    arch = parse_arch(args.config_file)
    ckpt_dir = args.ckpt_dir

    net = build_net(config, arch, device)
    ema_p = os.path.join(ckpt_dir, 'regressor_ema.pt')
    raw_p = os.path.join(ckpt_dir, 'regressor.pt')
    path = ema_p if (not args.no_ema and os.path.exists(ema_p)) else raw_p
    print(f"Loading weights: {path}")
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    net.eval()

    sigma = torch.load(os.path.join(ckpt_dir, 'regressor_scale.pt'),
                       map_location='cpu', weights_only=False)['sigma'].numpy().astype(np.float32)
    mean_np, std_np, mean_only = load_norm(config)

    template = load_mesh(config['template_file'], process=False)
    faces = np.array(template.faces)
    N_t = len(template.vertices)

    test_dir = args.test_dir or os.path.join(config['dataset_dir'],
                                             'completion_test_samples', 'test_set')
    out_dir = os.path.join(test_dir, 'completed_regressor')
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    print(f"Completing {len(gt_files)} test samples -> {out_dir}\n")

    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace('_gt_complete.ply', '')
        mask_file = os.path.join(test_dir, f'{base}_mask.npy')
        if not os.path.exists(mask_file):
            continue
        mask = np.load(mask_file).astype(bool)
        gt = np.array(load_mesh(gt_file, process=False).vertices, dtype=np.float32)
        if len(gt) != N_t:
            print(f"  skip {base}: size mismatch"); continue

        partial = gt.copy(); partial[mask] = 0.0
        vn = partial.astype(np.float32) - mean_np
        if not mean_only and std_np is not None:
            vn = vn / std_np
        vn[mask] = 0.0
        cond_xyz = vn / sigma; cond_xyz[mask] = 0.0
        mch = np.ones((N_t, 1), np.float32); mch[mask] = 0.0
        cond = torch.tensor(np.concatenate([cond_xyz, mch], 1), device=device)

        zeros = torch.zeros((N_t, 3), device=device)
        t0 = torch.zeros(1, device=device, dtype=torch.long)
        pred = net(zeros, cond, t0, 1).cpu().numpy() * sigma
        if not mean_only and std_np is not None:
            pred = pred * std_np
        pred = pred + mean_np

        comp = gt.copy(); comp[mask] = pred[mask]          # blend (known exact)
        Trimesh(vertices=comp, faces=faces, process=False).export(
            os.path.join(out_dir, f'{base}_completed_regressor.ply'))
        print(f"  {base}: missing {int(mask.sum())} done")

    print(f"\nDone. Score with research/eval_completion_dir.py on {out_dir}")


def main():
    ap = argparse.ArgumentParser(description='Deterministic regressor baseline (same backbone)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--config_file', required=True)
    pt.add_argument('--epochs', type=int, default=None)
    pt.add_argument('--batch_size', type=int, default=None)
    pt.add_argument('--ckpt_dir', default=DEFAULT_CKPT)
    pt.add_argument('--seed', type=int, default=1)

    pe = sub.add_parser('eval')
    pe.add_argument('--config_file', required=True)
    pe.add_argument('--ckpt_dir', default=DEFAULT_CKPT)
    pe.add_argument('--test_dir', default=None)
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
