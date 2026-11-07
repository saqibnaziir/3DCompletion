"""
Training script for mesh completion.

Key differences from train.py (reconstruction):
  - Uses MeshCompletionDataset (on-the-fly masking, 4-channel input)
  - Weighted loss: lambda_known * loss_known + lambda_missing * loss_missing
  - Optional transfer learning from pretrained reconstruction weights
  - Output is always 3-channel (xyz), input is 4-channel (xyz + mask)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import warnings
from tqdm import tqdm
from datetime import datetime

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset
from utils.models import FMGenModel, FMGenModelSummary
from utils.funcs import get_mesh_matrices, spherical_regularization_loss

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_model(config, pretrain=True):
    """
    Build the completion model (4-channel input, 3-channel output).
    Optionally initialize from pretrained reconstruction weights via transfer learning.
    """
    pA, pD, pU = get_mesh_matrices(config)
    model = FMGenModel(config, pA, pD, pU)

    if pretrain:
        encoder_path = config.get('pretrained_encoder', '')
        decoder_path = config.get('pretrained_decoder', '')

        # Resolve PATH_TO_PROJECT if present
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        encoder_path = encoder_path.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)
        decoder_path = decoder_path.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)

        if encoder_path and os.path.isfile(encoder_path):
            print(f"Loading pretrained encoder from: {encoder_path}")
            pretrained_enc = torch.load(encoder_path, map_location='cpu', weights_only=False)
            _transfer_weights(model.encoder.state_dict(), pretrained_enc)
            model.encoder.load_state_dict(model.encoder.state_dict())
            print("  Encoder weights transferred (3->4 channel adaptation applied where needed)")
        else:
            print("No pretrained encoder found, training from scratch.")

        if decoder_path and os.path.isfile(decoder_path):
            print(f"Loading pretrained decoder from: {decoder_path}")
            pretrained_dec = torch.load(decoder_path, map_location='cpu', weights_only=False)
            # Decoder is 3-channel output -> direct load (shapes should match)
            missing, unexpected = model.decoder.load_state_dict(pretrained_dec, strict=False)
            if missing:
                print(f"  Decoder missing keys: {missing}")
            if unexpected:
                print(f"  Decoder unexpected keys: {unexpected}")
            print("  Decoder weights loaded.")
        else:
            print("No pretrained decoder found, training decoder from scratch.")

    return model


def _transfer_weights(target_state_dict, source_state_dict):
    """
    Transfer weights from a 3-channel pretrained model to a 4-channel model.
    For layers where the input dimension changed (3 -> 4), copy the first 3 channels
    and randomly initialize the 4th (mask) channel.
    """
    for key in target_state_dict:
        if key in source_state_dict:
            src = source_state_dict[key]
            tgt = target_state_dict[key]
            if src.shape == tgt.shape:
                target_state_dict[key] = src
            elif len(src.shape) >= 1 and len(tgt.shape) >= 1:
                # Handle dimension mismatch (e.g., first conv layer weight)
                # Try to copy matching dimensions
                if src.shape[0] == tgt.shape[0] and len(src.shape) == 2 and len(tgt.shape) == 2:
                    # Linear layer: (out, in_old) -> (out, in_new)
                    min_in = min(src.shape[1], tgt.shape[1])
                    target_state_dict[key][:, :min_in] = src[:, :min_in]
                    print(f"  Adapted {key}: {src.shape} -> {tgt.shape}")
                elif src.shape[-1] == 3 and tgt.shape[-1] == 4:
                    # FeaStConv or similar: copy first 3 channels
                    target_state_dict[key][..., :3] = src
                    print(f"  Adapted {key}: {src.shape} -> {tgt.shape}")
                else:
                    print(f"  Skipped {key}: shape mismatch {src.shape} vs {tgt.shape}")
            else:
                print(f"  Skipped {key}: shape mismatch {src.shape} vs {tgt.shape}")


def completion_loss(output, target, vertex_mask, lambda_known, lambda_missing):
    """
    Weighted completion loss (L1 + MSE per region).

    Args:
        output: (B*N, 3) predicted complete vertices
        target: (B*N, 3) ground truth complete vertices
        vertex_mask: (B*N,) bool tensor, True = missing vertex
        lambda_known: weight for known (observed) vertices
        lambda_missing: weight for missing vertices

    Returns:
        loss_known, loss_missing, loss_total
    """
    known_mask = ~vertex_mask
    
    # L1 + MSE for known vertices
    if known_mask.any():
        loss_l1_known = F.l1_loss(output[known_mask], target[known_mask])
        loss_mse_known = F.mse_loss(output[known_mask], target[known_mask])
        loss_known = loss_l1_known + loss_mse_known
    else:
        loss_known = torch.tensor(0.0, device=output.device)

    # L1 + MSE for missing vertices
    if vertex_mask.any():
        loss_l1_missing = F.l1_loss(output[vertex_mask], target[vertex_mask])
        loss_mse_missing = F.mse_loss(output[vertex_mask], target[vertex_mask])
        loss_missing = loss_l1_missing + loss_mse_missing
    else:
        loss_missing = torch.tensor(0.0, device=output.device)

    loss_total = lambda_known * loss_known + lambda_missing * loss_missing
    return loss_known, loss_missing, loss_total


def chamfer_distance_loss(pred, target, vertex_mask, batch_idx):
    """
    Chamfer distance on missing vertices only.

    For each mesh in the batch, computes bidirectional nearest-neighbor
    distance between predicted and GT positions of missing vertices:

        CD = (1/M) * Σ_p min_q ||p - q||² + (1/M) * Σ_q min_p ||p - q||²

    This is more robust than per-vertex L2 for small spatial misalignments
    because it doesn't require exact vertex-to-vertex correspondence — if a
    predicted vertex is slightly shifted, it matches its nearest GT vertex.

    Args:
        pred: (B*N, 3) predicted vertices
        target: (B*N, 3) ground truth vertices
        vertex_mask: (B*N,) bool, True = missing
        batch_idx: (B*N,) int, mesh index for each vertex

    Returns:
        chamfer_loss: scalar tensor
    """
    device = pred.device
    B = batch_idx.max().item() + 1
    total_cd = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        b_mask = (batch_idx == b)
        missing = vertex_mask[b_mask]

        if missing.sum() < 2:
            continue

        p = pred[b_mask][missing]    # (M, 3) predicted missing vertices
        q = target[b_mask][missing]  # (M, 3) GT missing vertices

        # Pairwise squared distances: (M, M)
        # Using efficient computation: ||p-q||² = ||p||² + ||q||² - 2*p·q
        p_sq = (p ** 2).sum(dim=1, keepdim=True)   # (M, 1)
        q_sq = (q ** 2).sum(dim=1, keepdim=True)   # (M, 1)
        pq = torch.mm(p, q.t())                     # (M, M)
        dist2 = p_sq + q_sq.t() - 2.0 * pq          # (M, M)
        dist2 = dist2.clamp(min=0.0)                 # numerical safety

        # Forward Chamfer: for each predicted, find nearest GT
        min_p2q = dist2.min(dim=1)[0].mean()

        # Backward Chamfer: for each GT, find nearest predicted
        min_q2p = dist2.min(dim=0)[0].mean()

        total_cd = total_cd + min_p2q + min_q2p
        count += 1

    if count > 0:
        return total_cd / count
    return total_cd


def get_curriculum_mask_range(epoch, total_epochs, config):
    """
    Curriculum learning: linearly interpolate mask ratio from easy to hard.

    Early epochs: small masks (2-5%) — model learns basic geometry first.
    Late epochs:  larger masks (10-15%) — model learns harder completions.

    Returns:
        (mask_ratio_min, mask_ratio_max) for this epoch
    """
    progress = min(epoch / total_epochs, 1.0)

    start_min = config.get('curriculum_start_min', 0.02)
    start_max = config.get('curriculum_start_max', 0.05)
    end_min = config.get('curriculum_end_min', 0.10)
    end_max = config.get('curriculum_end_max', 0.15)

    cur_min = start_min + (end_min - start_min) * progress
    cur_max = start_max + (end_max - start_max) * progress

    return cur_min, cur_max


def train_epoch(model, train_loader, optimizer, device, dataset_size, epoch, writer, config):
    model.train()
    total_loss_known = 0
    total_loss_missing = 0
    total_loss_chamfer = 0
    total_loss_reg = 0
    total_loss = 0

    lambda_known = config.get('lambda_known', 1.0)
    lambda_missing = config.get('lambda_missing', 3.0)
    lambda_reg = config.get('lambda_reg', 0.001)
    use_chamfer = config.get('use_chamfer', False)
    lambda_chamfer = config.get('lambda_chamfer', 1.0)

    # Chamfer warm-up: linearly ramp from 0 to lambda_chamfer over first 20 epochs
    # This lets L1+MSE stabilize the geometry first, then Chamfer refines it
    chamfer_warmup_epochs = 20
    if use_chamfer and epoch <= chamfer_warmup_epochs:
        effective_lambda_chamfer = lambda_chamfer * (epoch / chamfer_warmup_epochs)
    else:
        effective_lambda_chamfer = lambda_chamfer

    for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # batch.x is (B*N, 4) = [partial_xyz, mask_channel]
        # batch.y is (B*N, 3) = complete xyz
        out, z = model(batch)

        # Gather vertex masks across the batch
        vertex_mask = batch.vertex_mask  # (B*N,) bool

        loss_known, loss_missing, loss_recon = completion_loss(
            out, batch.y, vertex_mask, lambda_known, lambda_missing
        )
        loss_reg = spherical_regularization_loss(z)

        loss = loss_recon + lambda_reg * loss_reg

        # V2: Chamfer distance on missing vertices (with warm-up)
        loss_cd = torch.tensor(0.0, device=device)
        if use_chamfer:
            loss_cd = chamfer_distance_loss(out, batch.y, vertex_mask, batch.batch)
            loss = loss + effective_lambda_chamfer * loss_cd

        loss.backward()
        optimizer.step()

        n = batch.num_graphs
        total_loss_known += n * loss_known.item()
        total_loss_missing += n * loss_missing.item()
        total_loss_chamfer += n * loss_cd.item()
        total_loss_reg += n * loss_reg.item()
        total_loss += n * loss.item()

    avg_known = total_loss_known / dataset_size
    avg_missing = total_loss_missing / dataset_size
    avg_chamfer = total_loss_chamfer / dataset_size
    avg_reg = total_loss_reg / dataset_size
    avg_total = total_loss / dataset_size

    writer.add_scalar('Loss/train/known', avg_known, epoch)
    writer.add_scalar('Loss/train/missing', avg_missing, epoch)
    if use_chamfer:
        writer.add_scalar('Loss/train/chamfer', avg_chamfer, epoch)
    writer.add_scalar('Loss/train/reg', avg_reg, epoch)
    writer.add_scalar('Loss/train/total', avg_total, epoch)

    return avg_known, avg_missing, avg_chamfer, avg_reg, avg_total


def eval_epoch(model, eval_loader, device, dataset_size, epoch, writer, config):
    model.eval()
    total_loss_known = 0
    total_loss_missing = 0
    total_loss_chamfer = 0
    total_loss_reg = 0
    total_loss = 0

    lambda_known = config.get('lambda_known', 1.0)
    lambda_missing = config.get('lambda_missing', 3.0)
    lambda_reg = config.get('lambda_reg', 0.001)
    use_chamfer = config.get('use_chamfer', False)
    lambda_chamfer = config.get('lambda_chamfer', 1.0)

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Eval epoch {epoch}"):
            batch = batch.to(device)

            out, z = model(batch)
            vertex_mask = batch.vertex_mask

            loss_known, loss_missing, loss_recon = completion_loss(
                out, batch.y, vertex_mask, lambda_known, lambda_missing
            )
            loss_reg = spherical_regularization_loss(z)
            loss = loss_recon + lambda_reg * loss_reg

            loss_cd = torch.tensor(0.0, device=device)
            if use_chamfer:
                loss_cd = chamfer_distance_loss(out, batch.y, vertex_mask, batch.batch)
                loss = loss + lambda_chamfer * loss_cd

            n = batch.num_graphs
            total_loss_known += n * loss_known.item()
            total_loss_missing += n * loss_missing.item()
            total_loss_chamfer += n * loss_cd.item()
            total_loss_reg += n * loss_reg.item()
            total_loss += n * loss.item()

    avg_known = total_loss_known / dataset_size
    avg_missing = total_loss_missing / dataset_size
    avg_chamfer = total_loss_chamfer / dataset_size
    avg_reg = total_loss_reg / dataset_size
    avg_total = total_loss / dataset_size

    writer.add_scalar('Loss/eval/known', avg_known, epoch)
    writer.add_scalar('Loss/eval/missing', avg_missing, epoch)
    if use_chamfer:
        writer.add_scalar('Loss/eval/chamfer', avg_chamfer, epoch)
    writer.add_scalar('Loss/eval/reg', avg_reg, epoch)
    writer.add_scalar('Loss/eval/total', avg_total, epoch)

    return avg_known, avg_missing, avg_chamfer, avg_reg, avg_total


def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_chamfer = config.get('use_chamfer', False)
    use_curriculum = config.get('curriculum', False)

    # TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['experiment_name']}_{current_time}"
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard_logs', run_name))
    print(f"TensorBoard logs: {writer.log_dir}")

    # Print V2 features
    if use_chamfer:
        print(f"V2: Chamfer distance ENABLED (lambda_chamfer={config.get('lambda_chamfer', 1.0)})")
    if use_curriculum:
        print(f"V2: Curriculum learning ENABLED")
        print(f"     Start: [{config.get('curriculum_start_min', 0.02):.2f}, "
              f"{config.get('curriculum_start_max', 0.05):.2f}]")
        print(f"     End:   [{config.get('curriculum_end_min', 0.10):.2f}, "
              f"{config.get('curriculum_end_max', 0.15):.2f}]")

    # Datasets
    print("Loading completion datasets...")
    dataset_train = MeshCompletionDataset(config, 'train')
    dataset_eval = MeshCompletionDataset(config, 'eval')

    # With curriculum learning, we use num_workers=0 to allow dynamic mask ratio updates.
    # (Worker processes get a copy of the dataset at fork time, so attribute updates
    #  on the main process won't propagate to persistent workers.)
    if use_curriculum:
        num_workers = 0
        persistent_workers = False
        print("  Using num_workers=0 for curriculum learning (required for dynamic mask ratio)")
    else:
        num_workers = config['num_workers']
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers
    )
    eval_loader = DataLoader(
        dataset_eval, batch_size=config['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers
    )

    # Model
    print("Loading model...")
    pretrain = bool(config.get('pretrained_encoder', ''))
    model = load_model(config, pretrain=pretrain)
    model.to(device)

    # Print summary
    summary = FMGenModelSummary(model, config)
    sample_batch = next(iter(train_loader)).to(device)
    summary.print_summary(sample_batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print("Starting completion training...")
    best_eval_loss = float('inf')

    # Curriculum learning: fix eval mask ratio at the FINAL target difficulty
    # so that eval loss is comparable across all epochs.
    # Only the training mask ratio changes (easy -> hard).
    if use_curriculum:
        eval_min = config.get('curriculum_end_min', 0.10)
        eval_max = config.get('curriculum_end_max', 0.15)
        dataset_eval.set_mask_ratio_range(eval_min, eval_max)
        print(f"  Eval mask ratio FIXED at [{eval_min:.3f}, {eval_max:.3f}] (final target difficulty)")

    for epoch in range(1, config['epoch'] + 1):
        # Curriculum learning: update TRAINING mask ratio only
        if use_curriculum:
            cur_min, cur_max = get_curriculum_mask_range(epoch, config['epoch'], config)
            dataset_train.set_mask_ratio_range(cur_min, cur_max)
            writer.add_scalar('Curriculum/mask_min', cur_min, epoch)
            writer.add_scalar('Curriculum/mask_max', cur_max, epoch)
            print(f"\nEpoch {epoch}  lr: {scheduler.get_last_lr()}  "
                  f"train_mask: [{cur_min:.3f}, {cur_max:.3f}]  "
                  f"eval_mask: [{eval_min:.3f}, {eval_max:.3f}]")
        else:
            print(f"\nEpoch {epoch}  lr: {scheduler.get_last_lr()}")

        train_known, train_missing, train_chamfer, train_reg, train_total = train_epoch(
            model, train_loader, optimizer, device, len(dataset_train), epoch, writer, config
        )
        if use_chamfer:
            chamfer_warmup = min(epoch / 20.0, 1.0)
            chamfer_str = f"  chamfer: {train_chamfer:.6f} (w={chamfer_warmup:.2f})"
            writer.add_scalar('Chamfer/warmup_weight', chamfer_warmup, epoch)
        else:
            chamfer_str = ""
        print(f"  Train  | known: {train_known:.6f}  missing: {train_missing:.6f}"
              f"{chamfer_str}  reg: {train_reg:.6f}  total: {train_total:.6f}")

        eval_known, eval_missing, eval_chamfer, eval_reg, eval_total = eval_epoch(
            model, eval_loader, device, len(dataset_eval), epoch, writer, config
        )
        chamfer_str = f"  chamfer: {eval_chamfer:.6f}" if use_chamfer else ""
        print(f"  Eval   | known: {eval_known:.6f}  missing: {eval_missing:.6f}"
              f"{chamfer_str}  reg: {eval_reg:.6f}  total: {eval_total:.6f}")

        scheduler.step()
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        # Save best model
        if eval_total < best_eval_loss:
            best_eval_loss = eval_total
            torch.save(model.encoder.state_dict(),
                       os.path.join(checkpoint_dir, 'completion_encoder.pt'))
            torch.save(model.decoder.state_dict(),
                       os.path.join(checkpoint_dir, 'completion_decoder.pt'))
            torch.save(scheduler.state_dict(),
                       os.path.join(checkpoint_dir, 'completion_scheduler.pt'))
            print(f"  ** Saved best model (eval_total={eval_total:.6f}) **")

    writer.close()
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description='Train mesh completion model')
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to config_completion.cfg")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name")
    parser.add_argument("--model", type=str, default=None,
                        choices=['default', 'global_only', 'local_only', 'gat', 'gcn'],
                        help="Model ablation variant")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_pretrain", action='store_true',
                        help="Train from scratch (ignore pretrained weights)")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = read_config(args.config_file)

    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.model is not None:
        config['model_type'] = args.model
    if args.no_pretrain:
        config['pretrained_encoder'] = ''
        config['pretrained_decoder'] = ''

    config['seed'] = args.seed
    config['experiment_name'] = f"{config['experiment_name']}_seed{args.seed}"

    train(config)


if __name__ == "__main__":
    main()
