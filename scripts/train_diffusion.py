"""
Training script for diffusion-based mesh completion (Option B).

Diffuses the per-vertex offset field (== normalized complete mesh) conditioned
on a partial observation, using a graph U-Net denoiser. Reuses the existing
MeshCompletionDataset (on-the-fly masking, 4-channel conditioning).

Run:
  python scripts/train_diffusion.py --config_file config/config_diffusion.cfg

RESUME-SAFE: a full state (model+ema+optimizer+scheduler+epoch+best) is written
every epoch to diffusion_state.pt, plus a per-epoch line to train_log.csv, so a
reboot mid-run costs at most one epoch -- just relaunch the SAME command with
--resume added.

Runs started before diffusion_state.pt existed can still be picked up from the
weight-only checkpoints with --resume_from_weights --start_epoch N (the Adam
moments are lost, so expect a short re-warmup).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import configparser
import csv
import glob
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import read_config, resolve_config_file
from utils.completion_dataset import MeshCompletionDataset
from utils.funcs import get_mesh_matrices
from utils.diffusion import GaussianDiffusion, EMA
from utils.diffusion_model import GraphUNetDenoiser
from utils.normal_loss import (batched_faces, normal_cosine_loss,
                               normal_angle_degrees)

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ----------------------------------------------------------------------------
def parse_diffusion_section(config_file):
    """Read the [diffusion parameters] block (not handled by read_config)."""
    cp = configparser.RawConfigParser()
    cp.read(config_file)
    d = {}
    sec = 'diffusion parameters'
    d['timesteps'] = cp.getint(sec, 'timesteps', fallback=1000)
    d['schedule'] = cp.get(sec, 'schedule', fallback='cosine')
    d['predict'] = cp.get(sec, 'predict', fallback='eps')
    d['base_channels'] = cp.getint(sec, 'base_channels', fallback=64)
    d['channel_mults'] = tuple(int(x) for x in
                               cp.get(sec, 'channel_mults', fallback='1,2,4,4').split(','))
    d['time_embed_dim'] = cp.getint(sec, 'time_embed_dim', fallback=128)
    d['ema_decay'] = cp.getfloat(sec, 'ema_decay', fallback=0.9995)
    d['sampling_steps'] = cp.getint(sec, 'sampling_steps', fallback=50)
    d['sampling_clip'] = cp.getfloat(sec, 'sampling_clip', fallback=6.0)
    d['lambda_laplacian'] = cp.getfloat(sec, 'lambda_laplacian', fallback=0.0)
    # Surface-normal loss (Olivier). 0.0 = OFF, which is the default everywhere,
    # so eps/x0/depthw runs reproduce byte-for-byte.
    d['lambda_normal'] = cp.getfloat(sec, 'lambda_normal', fallback=0.0)
    d['normal_snr_weight'] = cp.getint(sec, 'normal_snr_weight', fallback=1) == 1
    d['normal_use_depth_weight'] = cp.getint(sec, 'normal_use_depth_weight',
                                             fallback=0) == 1
    return d


def laplacian_loss(verts, edge_index, missing_mask):
    """Mean-squared discrete Laplacian on the missing region (uniform weights)."""
    row, col = edge_index[0], edge_index[1]
    M = verts.shape[0]
    deg = torch.zeros(M, device=verts.device).index_add_(
        0, col, torch.ones_like(col, dtype=verts.dtype))
    nsum = torch.zeros_like(verts).index_add_(0, col, verts[row])
    nmean = nsum / deg.clamp(min=1).unsqueeze(-1)
    lap = verts - nmean
    if missing_mask.any():
        return (lap[missing_mask] ** 2).mean()
    return torch.tensor(0.0, device=verts.device)


# ----------------------------------------------------------------------------
def run_epoch(denoiser, diffusion, loader, optimizer, device, N, dcfg,
              lambda_known, lambda_missing, sigma, ema=None, train=True,
              faces=None):
    """
    Returns (mean_total_loss, mean_normal_deg). The second value is NaN unless
    the normal loss is active; it is the mean angular error of the predicted x0
    in degrees, logged so the run is readable in the same units as
    compare_arms.py without putting arccos in the gradient path.
    """
    denoiser.train(train)
    total = 0.0
    n_seen = 0
    use_lap = dcfg['lambda_laplacian'] > 0.0
    use_nrm = dcfg.get('lambda_normal', 0.0) > 0.0 and faces is not None
    nrm_deg_sum = 0.0
    faces_cache = {}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc='train' if train else 'eval', leave=False):
            batch = batch.to(device)
            B = batch.num_graphs
            # Standardize the diffusion target to ~unit variance (sigma is per-axis
            # std of the offset field). Without this the signal is ~300x smaller
            # than the noise and the model cannot learn / sampling diverges.
            x0 = batch.y / sigma               # (B*N, 3) standardized offsets
            cond = batch.x.clone()             # (B*N, 4) [partial_xyz, mask]
            cond[:, :3] = cond[:, :3] / sigma  # condition lives in the same space
            vmask = batch.vertex_mask          # (B*N,) True = missing

            # Per-vertex geodesic depth weight, if the dataset produced one.
            vw = getattr(batch, 'loss_weight', None)

            loss, model_out, t, x_t, noise = diffusion.p_losses(
                denoiser, x0, cond, B, N, vmask,
                lambda_known=lambda_known, lambda_missing=lambda_missing,
                vertex_weight=vw)

            if use_lap or use_nrm:
                # Decode once and share: both extra terms act on the predicted
                # clean mesh, not on the raw network output.
                t_node = t.repeat_interleave(N)
                x0_pred, _ = diffusion.decode(model_out, x_t, t_node)

                if use_lap:
                    ei0 = denoiser._batched_edge_index(0, B, device)
                    loss = loss + dcfg['lambda_laplacian'] * laplacian_loss(
                        x0_pred, ei0, vmask)

                if use_nrm:
                    if B not in faces_cache:
                        faces_cache[B] = batched_faces(faces, B, N, device)
                    fb = faces_cache[B]
                    # alpha_bar_t: 1 -> 0 as noise grows, so the term is applied
                    # where the predicted x0 is actually informative.
                    snr = (diffusion.alphas_cumprod[t_node]
                           if dcfg['normal_snr_weight'] else None)
                    n_loss = normal_cosine_loss(
                        x0_pred, x0, fb, sigma,
                        region_mask=vmask, snr_weight=snr,
                        vertex_weight=vw if dcfg['normal_use_depth_weight'] else None)
                    loss = loss + dcfg['lambda_normal'] * n_loss
                    # Log with the SAME alpha_bar weighting the loss uses --
                    # an unweighted mean is dominated by high-noise steps and is
                    # flat regardless of progress (see normal_angle_degrees).
                    nrm_deg_sum += B * float(normal_angle_degrees(
                        x0_pred, x0, fb, sigma, region_mask=vmask, weight=snr))

                    if train and not getattr(run_epoch, '_balance_printed', False):
                        run_epoch._balance_printed = True
                        # One-time readout so lambda_normal can be set from a
                        # measurement instead of a guess: aim for the weighted
                        # normal term at ~0.1-0.3x the positional term.
                        pos_term = float(loss) - dcfg['lambda_normal'] * float(n_loss)
                        print(f"\n  [normal loss] first batch: position={pos_term:.6f}  "
                              f"normal={float(n_loss):.6f}  "
                              f"lambda*normal={dcfg['lambda_normal']*float(n_loss):.6f} "
                              f"({dcfg['lambda_normal']*float(n_loss)/max(pos_term,1e-12):.2f}x "
                              f"position)")

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(denoiser)

            total += B * loss.item()
            n_seen += B

    n = max(n_seen, 1)
    return total / n, (nrm_deg_sum / n if use_nrm else float('nan'))


def train(config, dcfg, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_dir = config['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, 'diffusion_state.pt')
    log_path = os.path.join(ckpt_dir, 'train_log.csv')

    resuming = args.resume or args.resume_from_weights
    log_root = os.path.join(ckpt_dir, 'tensorboard_logs')
    prior_runs = sorted(glob.glob(os.path.join(log_root, f"{config['experiment_name']}_*")))
    if resuming and prior_runs:
        # Append to the existing run so the curves stay continuous (epoch numbers
        # do not overlap, so TensorBoard stitches them into one line).
        log_dir = prior_runs[-1]
    else:
        log_dir = os.path.join(
            log_root, f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {writer.log_dir}")

    print("Loading datasets...")
    ds_train = MeshCompletionDataset(config, 'train')
    ds_eval = MeshCompletionDataset(config, 'eval')
    N = ds_train.get(0).x.shape[0]            # vertices per mesh (fixed)
    print(f"Vertices per mesh: {N}  |  train={len(ds_train)} eval={len(ds_eval)}")

    # Per-axis std of the offset field -> standardize the diffusion target to
    # unit variance. Also store a clip value (max standardized magnitude) used to
    # stabilize DDIM sampling. Saved so inference uses the exact same values.
    sigma = ds_train.data.y.std(dim=0).clamp(min=1e-6).to(device)
    std_absmax = float((ds_train.data.y / sigma.cpu()).abs().max())
    clip_value = dcfg.get('sampling_clip', 6.0)          # empirical sweet spot
    torch.save({'sigma': sigma.cpu(), 'clip_value': clip_value},
               os.path.join(ckpt_dir, 'diffusion_scale.pt'))
    print(f"Diffusion target std (sigma): {sigma.tolist()}")
    print(f"Data standardized absmax: {std_absmax:.2f}  |  saved clip_value: {clip_value}")

    nw = config['num_workers']
    train_loader = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)
    eval_loader = DataLoader(ds_eval, batch_size=config['batch_size'], shuffle=False,
                             num_workers=nw, pin_memory=True, drop_last=True,
                             persistent_workers=nw > 0)

    print("Building denoiser + diffusion...")
    pA, pD, pU = get_mesh_matrices(config)
    denoiser = GraphUNetDenoiser(
        list(pA), list(pD), list(pU),
        in_ch=7, out_ch=3,
        base_channels=dcfg['base_channels'],
        channel_mults=dcfg['channel_mults'],
        time_embed_dim=dcfg['time_embed_dim'],
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=dcfg['timesteps'],
                                  schedule=dcfg['schedule'],
                                  predict=dcfg['predict']).to(device)
    print(f"Prediction target: {dcfg['predict']}")
    ema = EMA(denoiser, decay=dcfg['ema_decay'])

    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"Denoiser parameters: {n_params:,}")
    print(f"Timesteps={dcfg['timesteps']} schedule={dcfg['schedule']} "
          f"channels={[dcfg['base_channels']*m for m in dcfg['channel_mults']]}")

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    # Cosine LR decay over the full run (helps diffusion final quality).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epoch'], eta_min=config['lr'] * 0.05)

    # ---- resume -------------------------------------------------------------
    start_epoch, best_eval = 1, float('inf')
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        denoiser.load_state_dict(st['model'])
        ema.load_state_dict(st['ema'])
        optimizer.load_state_dict(st['optimizer'])
        scheduler.load_state_dict(st['scheduler'])
        start_epoch = st['epoch'] + 1
        best_eval = st['best_eval']
        print(f"** resumed from epoch {st['epoch']} (best eval={best_eval:.6f}) "
              f"-> continuing at epoch {start_epoch} **")
    elif args.resume_from_weights:
        # Bootstrap for runs that predate diffusion_state.pt: only the weight
        # files exist, so the Adam moments are gone and the scheduler has to be
        # fast-forwarded by hand.
        if args.start_epoch is None:
            raise SystemExit("--resume_from_weights requires --start_epoch "
                             "(the epoch the saved weights come from, + 1)")
        raw_p = os.path.join(ckpt_dir, 'diffusion_denoiser.pt')
        ema_p = os.path.join(ckpt_dir, 'diffusion_denoiser_ema.pt')
        denoiser.load_state_dict(torch.load(raw_p, map_location=device, weights_only=False))
        ema.load_state_dict(torch.load(ema_p, map_location=device, weights_only=False))
        start_epoch = args.start_epoch
        best_eval = args.best_eval
        for _ in range(start_epoch - 1):       # replay the cosine decay
            scheduler.step()
        print(f"** bootstrapped from weights {raw_p}\n"
              f"   -> continuing at epoch {start_epoch}, lr {scheduler.get_last_lr()[0]:.3e}, "
              f"best_eval={best_eval:.6f}\n"
              f"   NOTE: optimizer state was not saved by the old run; Adam needs a "
              f"short re-warmup. **")
    elif args.resume:
        print(f"--resume given but no {state_path}; starting fresh.")

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train', 'eval', 'lr', 'timestamp'])

    lambda_known = config.get('lambda_known', 1.0)
    lambda_missing = config.get('lambda_missing', 3.0)

    if config.get('depth_weight', False):
        print(f"Loss weighting: GEODESIC DEPTH RAMP  w = {lambda_known} + "
              f"({config['depth_weight_max']} - {lambda_known}) * d/d_max  on missing, "
              f"{lambda_known} on known\n"
              f"  backend={config.get('depth_weight_backend', 'dijkstra')}  "
              f"normalize={config.get('depth_weight_normalize', False)}")
        if not config.get('depth_weight_normalize', False):
            print("  NOTE: the raw ramp averages ~w_max/2 over the hole, so the mean "
                  "missing-region\n        weight is ~half the flat baseline's. Loss "
                  "values are NOT comparable across arms;\n        compare on test-set "
                  "missing-region error instead.")
    else:
        print(f"Loss weighting: FLAT  known={lambda_known}  missing={lambda_missing}")

    # Template faces for the surface-normal loss. Loaded once; topology is fixed,
    # so the same face list serves every sample and every batch.
    faces_t = None
    if dcfg.get('lambda_normal', 0.0) > 0.0:
        from trimesh import load_mesh as _load_mesh
        tpl = _load_mesh(config['template_file'], process=False)
        faces_t = torch.as_tensor(np.asarray(tpl.faces), dtype=torch.long, device=device)
        if len(tpl.vertices) != N:
            raise SystemExit(
                f"Template has {len(tpl.vertices)} vertices but the dataset has {N}. "
                f"The normal loss needs them to match.")
        print(f"Normal loss: ON  lambda_normal={dcfg['lambda_normal']}  "
              f"snr_weight={dcfg['normal_snr_weight']}  "
              f"depth_weighted={dcfg['normal_use_depth_weight']}\n"
              f"  {len(faces_t)} faces | region=missing | un-standardized by sigma "
              f"before taking normals")
    else:
        print("Normal loss: OFF")

    print(f"Starting diffusion training (epochs {start_epoch}..{config['epoch']})")
    for epoch in range(start_epoch, config['epoch'] + 1):
        tr, tr_deg = run_epoch(denoiser, diffusion, train_loader, optimizer, device,
                               N, dcfg, lambda_known, lambda_missing, sigma,
                               ema=ema, train=True, faces=faces_t)
        ev, ev_deg = run_epoch(denoiser, diffusion, eval_loader, optimizer, device,
                               N, dcfg, lambda_known, lambda_missing, sigma,
                               ema=None, train=False, faces=faces_t)

        lr_now = scheduler.get_last_lr()[0]
        writer.add_scalar('Loss/train', tr, epoch)
        writer.add_scalar('Loss/eval', ev, epoch)
        writer.add_scalar('LR', lr_now, epoch)
        deg_str = ""
        if ev_deg == ev_deg:                       # not NaN => normal loss active
            writer.add_scalar('NormalDeg/train', tr_deg, epoch)
            writer.add_scalar('NormalDeg/eval', ev_deg, epoch)
            deg_str = f" | normals {tr_deg:.3f}/{ev_deg:.3f} deg"
        print(f"Epoch {epoch:4d} | train {tr:.6f} | eval {ev:.6f} | "
              f"lr {lr_now:.2e}{deg_str}")
        scheduler.step()

        if ev < best_eval:
            best_eval = ev
            torch.save(denoiser.state_dict(), os.path.join(ckpt_dir, 'diffusion_denoiser.pt'))
            torch.save(ema.state_dict(), os.path.join(ckpt_dir, 'diffusion_denoiser_ema.pt'))
            print(f"  ** saved best (eval={ev:.6f}) **")

        # Always keep a 'last' EMA snapshot too (sampling uses EMA).
        torch.save(ema.state_dict(), os.path.join(ckpt_dir, 'diffusion_denoiser_ema_last.pt'))

        # Full resumable state, written every epoch. Save to a temp file first so
        # a crash mid-write cannot leave a truncated state behind.
        tmp = state_path + '.tmp'
        torch.save({'model': denoiser.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_eval': best_eval}, tmp)
        os.replace(tmp, state_path)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{tr:.6f}", f"{ev:.6f}", f"{lr_now:.6e}",
                                    datetime.now().isoformat(timespec='seconds')])

    writer.close()
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description='Train diffusion mesh completion')
    parser.add_argument("--config_file", type=str, default=None,
                        help="explicit config path (overrides --data)")
    parser.add_argument("--data", type=str, default=None, choices=['COMA', 'DFAUST'],
                        help="dataset preset: selects config/config_diffusion[_dfaust].cfg")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--param", type=str, default=None, choices=['eps', 'x0', 'v'],
                        help="prediction target; overrides [diffusion parameters] predict")
    parser.add_argument("--depth_weight", action='store_true',
                        help="geodesic depth-ramped loss weight: w = 1 + (w_max-1)*d/d_max "
                             "over the missing region (d = geodesic distance to the known "
                             "boundary), instead of a flat lambda_missing")
    parser.add_argument("--depth_weight_max", type=float, default=None,
                        help="w_max for --depth_weight (default: lambda_missing)")
    parser.add_argument("--depth_weight_normalize", action='store_true',
                        help="rescale the ramp so its mean over the hole equals w_max "
                             "(isolates the SHAPE of the weighting from the overall "
                             "missing-region emphasis)")
    parser.add_argument("--depth_weight_backend", type=str, default=None,
                        choices=['dijkstra', 'hops', 'euclidean', 'heat'],
                        help="how to measure d(v) (default dijkstra: exact on the edge graph)")
    parser.add_argument("--lambda_normal", type=float, default=None,
                        help="weight of the surface-normal loss (1 - cos between "
                             "predicted and GT vertex normals over the missing region). "
                             "0 = off (default). Start from the one-time 'first batch' "
                             "readout printed at startup: aim for ~0.1-0.3x the "
                             "positional term")
    parser.add_argument("--no_normal_snr", action='store_true',
                        help="do NOT weight the normal loss by alpha_bar_t. Off by "
                             "default because at high noise the predicted x0 is "
                             "meaningless and its normals are worse than meaningless")
    parser.add_argument("--normal_use_depth_weight", action='store_true',
                        help="also apply the geodesic depth ramp to the normal term "
                             "(default: uniform over the hole)")
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for the checkpoint subfolder, e.g. --tag depthw -> "
                             "predict_x0_depthw (keeps arms from overwriting each other)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action='store_true',
                        help="continue from diffusion_state.pt in the arm's checkpoint "
                             "dir if present (model+ema+optimizer+scheduler+epoch)")
    parser.add_argument("--resume_from_weights", action='store_true',
                        help="bootstrap a run that predates diffusion_state.pt: load "
                             "diffusion_denoiser.pt + _ema.pt and fast-forward the LR "
                             "schedule. Requires --start_epoch; optimizer state is lost")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="first epoch to run when using --resume_from_weights "
                             "(i.e. the saved weights' epoch + 1)")
    parser.add_argument("--best_eval", type=float, default=float('inf'),
                        help="best eval loss so far, for --resume_from_weights. Pass it, "
                             "or the first epoch will overwrite the best checkpoint")
    args = parser.parse_args()

    if args.resume and args.resume_from_weights:
        parser.error("--resume and --resume_from_weights are mutually exclusive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    cfg_path = resolve_config_file(args.data, args.config_file)
    print(f"Using config: {cfg_path}")
    config = read_config(cfg_path)
    dcfg = parse_diffusion_section(cfg_path)

    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.epochs is not None:
        config['epoch'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.param is not None:
        dcfg['predict'] = args.param
    if args.depth_weight:
        config['depth_weight'] = True
    if args.depth_weight_max is not None:
        config['depth_weight_max'] = args.depth_weight_max
    if args.depth_weight_normalize:
        config['depth_weight_normalize'] = True
    if args.depth_weight_backend is not None:
        config['depth_weight_backend'] = args.depth_weight_backend
    if args.lambda_normal is not None:
        dcfg['lambda_normal'] = args.lambda_normal
    if args.no_normal_snr:
        dcfg['normal_snr_weight'] = False
    if args.normal_use_depth_weight:
        dcfg['normal_use_depth_weight'] = True
    # depth_weight_max defaults to lambda_missing so the ramp tops out exactly
    # where the flat baseline sat.
    config.setdefault('depth_weight_max', config.get('lambda_missing', 3.0))

    # Isolate each ablation in its own checkpoint subfolder so the arms don't
    # overwrite each other's weights + scale file.
    arm = f"predict_{dcfg['predict']}"
    if args.tag:
        arm = f"{arm}_{args.tag}"
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], arm)
    config['experiment_name'] = f"{config['experiment_name']}_{arm}_seed{args.seed}"

    train(config, dcfg, args)


if __name__ == "__main__":
    main()
