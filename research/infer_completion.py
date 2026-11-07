"""
Inference script for mesh completion.

Given a partial mesh (with missing/zeroed regions), runs a single forward pass
through the trained completion model and outputs the completed mesh.

Also supports batch evaluation on the saved test samples (partial + GT pairs)
with per-vertex error computation.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
from glob import glob
from trimesh import Trimesh, load_mesh
from torch_geometric.data import Data, Batch

from config.config import read_config
from utils.funcs import get_mesh_matrices, save_ply_explicit, get_edge_index
from utils.models import FMGenModel
from utils.postprocess import postprocess_completion


def load_completion_model(config, device):
    """Load the trained completion model."""
    pA, pD, pU = get_mesh_matrices(config)
    model = FMGenModel(config, pA, pD, pU)

    checkpoint_dir = config['checkpoint_dir']
    enc_path = os.path.join(checkpoint_dir, 'completion_encoder.pt')
    dec_path = os.path.join(checkpoint_dir, 'completion_decoder.pt')

    model.encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=False))
    model.decoder.load_state_dict(torch.load(dec_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded completion model from {checkpoint_dir}")
    return model


def load_norm(config):
    """Load normalization parameters for the completion dataset."""
    norm_path = os.path.join(config['dataset_dir'], 'norm_completion.pt')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(config['dataset_dir'], 'norm.pt')
    norm_dict = torch.load(norm_path, map_location='cpu', weights_only=False)
    mean = norm_dict['mean']
    std = norm_dict['std']
    mean_subtraction_only = std is None
    return mean, std, mean_subtraction_only


def get_template_info(config):
    """Load template mesh and edge_index (cached)."""
    template_mesh = load_mesh(config['template_file'], process=False)
    edge_index_path = os.path.join(config['dataset_dir'], 'processed_completion', 'edge_index.pt')
    if os.path.exists(edge_index_path):
        edge_index = torch.load(edge_index_path, weights_only=False)
    else:
        edge_index = get_edge_index(template_mesh.vertices, template_mesh.faces)
    return template_mesh, edge_index


def _run_forward(model, verts_np, missing_mask, config, device, template_mesh, edge_index):
    """
    Core forward pass: takes vertices (N,3) and missing_mask (N,) bool,
    builds 4-ch input, runs model, returns denormalized output (N,3).

    The vertex count MUST match the template (model's expected topology).
    """
    N = len(verts_np)
    N_template = len(template_mesh.vertices)
    assert N == N_template, (
        f"Vertex count mismatch: input has {N} vertices but model expects {N_template}. "
        f"The input mesh must have the same topology as the template."
    )

    # Normalize
    mean, std, mean_subtraction_only = load_norm(config)
    mean_np = mean.numpy()
    std_np = std.numpy() if std is not None else None

    verts_norm = verts_np.astype(np.float32).copy() - mean_np
    if not mean_subtraction_only and std_np is not None:
        verts_norm = verts_norm / std_np
    verts_norm[missing_mask] = 0.0  # zero out missing vertices

    # Build 4-channel input: [x, y, z, mask_channel]
    mask_channel = np.ones((N, 1), dtype=np.float32)
    mask_channel[missing_mask] = 0.0
    input_features = np.concatenate([verts_norm, mask_channel], axis=1)  # (N, 4)

    input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    data = Data(x=input_tensor, edge_index=edge_index.to(device))
    batch = Batch.from_data_list([data])

    # Forward pass
    with torch.no_grad():
        out, z = model(batch)

    # Denormalize
    out_np = out.cpu().numpy()
    if not mean_subtraction_only and std_np is not None:
        out_np = out_np * std_np
    out_np = out_np + mean_np

    return out_np


def complete_from_vertices(model, verts_np, missing_mask, config, device,
                           template_mesh, edge_index, output_path=None,
                           smooth=True, verbose=True):
    """
    Complete a mesh given raw vertices and a boolean missing mask.

    Args:
        verts_np: (N, 3) numpy array of vertex positions (missing = zeroed)
        missing_mask: (N,) bool array, True = missing vertex
        output_path: optional path to save the completed .ply
        smooth: apply boundary-aware Taubin smoothing as post-processing

    Returns:
        final_verts: (N, 3) completed vertex positions
    """
    if verbose:
        num_missing = missing_mask.sum()
        print(f"  Missing: {num_missing}/{len(verts_np)} vertices "
              f"({100 * num_missing / len(verts_np):.1f}%)")

    out_np = _run_forward(model, verts_np, missing_mask, config, device, template_mesh, edge_index)

    # Blend: keep original known vertices, use model prediction for missing
    final_verts = verts_np.copy()
    final_verts[missing_mask] = out_np[missing_mask]

    # Post-processing: boundary-aware smoothing to remove seam artifacts
    if smooth:
        faces_np = np.array(template_mesh.faces)
        final_verts, num_smoothed = postprocess_completion(
            final_verts, faces_np, missing_mask,
            method='taubin', num_rings=3, iterations=5,
            lambda_factor=0.5, mu_factor=-0.53
        )
        if verbose:
            print(f"  Post-processed: Taubin smoothing on {num_smoothed} boundary vertices")

    if output_path is not None:
        completed_mesh = Trimesh(vertices=final_verts, faces=template_mesh.faces, process=False)
        save_ply_explicit(completed_mesh, output_path)
        if verbose:
            print(f"  Saved completed mesh to {output_path}")

    return final_verts


def complete_single_mesh(model, partial_mesh_path, output_path, config, device,
                         template_mesh, edge_index, smooth=True, verbose=True):
    """
    Complete a single partial mesh file.
    The mesh MUST have the same vertex count/topology as the template.
    Use process=False when loading to prevent trimesh from altering vertices.
    """
    # Load with process=False to preserve all vertices (including zeroed ones)
    partial_mesh = load_mesh(partial_mesh_path, process=False)
    verts = np.array(partial_mesh.vertices, dtype=np.float32)

    N_template = len(template_mesh.vertices)
    if len(verts) != N_template:
        if verbose:
            print(f"  WARNING: {partial_mesh_path} has {len(verts)} vertices, "
                  f"expected {N_template}. Skipping.")
        return None, None

    # Detect missing vertices (zeroed out)
    missing_mask = np.linalg.norm(verts, axis=1) < 1e-6

    return complete_from_vertices(
        model, verts, missing_mask, config, device,
        template_mesh, edge_index, output_path,
        smooth=smooth, verbose=verbose
    )


def evaluate_test_samples(model, config, device, template_mesh, edge_index,
                          smooth=True, verbose=True, test_dir=None):
    """
    Evaluate on the saved test samples (GT + mask).
    Saves THREE versions for comparison:
      1. Raw (model output, no post-processing)
      2. Stitched (boundary stitching only)
      3. Stitched + Taubin (boundary stitching followed by light Taubin smoothing)
    """
    if test_dir is None:
        test_dir = os.path.join(config['dataset_dir'], 'completion_test_samples')
    if not os.path.exists(test_dir):
        print(f"No test samples found in {test_dir}. Run dataset processing first.")
        return

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    print(f"\nEvaluating {len(gt_files)} test samples (saving raw + stitched + stitch+taubin)...")

    output_dir = os.path.join(test_dir, 'completed')
    os.makedirs(output_dir, exist_ok=True)

    faces_np = np.array(template_mesh.faces)

    for gt_file in gt_files:
        basename = os.path.basename(gt_file).replace('_gt_complete.ply', '')

        mask_file = os.path.join(test_dir, f'{basename}_mask.npy')
        if not os.path.exists(mask_file):
            print(f"  Skipping {basename}: no mask file found")
            continue
        mask = np.load(mask_file)

        gt_mesh = load_mesh(gt_file, process=False)
        gt_verts = np.array(gt_mesh.vertices, dtype=np.float32)

        N_template = len(template_mesh.vertices)
        if len(gt_verts) != N_template:
            print(f"  Skipping {basename}: GT has {len(gt_verts)} verts, expected {N_template}")
            continue
        if len(mask) != len(gt_verts):
            print(f"  Skipping {basename}: mask size mismatch")
            continue

        # Build partial input
        partial_verts = gt_verts.copy()
        partial_verts[mask] = 0.0

        # --- 1) Raw completion (no post-processing) ---
        raw_file = os.path.join(output_dir, f'{basename}_completed_raw.ply')
        raw_verts = complete_from_vertices(
            model, partial_verts, mask, config, device,
            template_mesh, edge_index, output_path=raw_file,
            smooth=False, verbose=False
        )

        # --- 2) Boundary stitching only ---
        stitch_file = os.path.join(output_dir, f'{basename}_completed_stitch.ply')
        stitch_verts = raw_verts.copy()
        stitch_verts, num_stitched = postprocess_completion(
            stitch_verts, faces_np, mask,
            method='stitch', num_rings=3, stitch_weight=0.7
        )
        stitch_mesh = Trimesh(vertices=stitch_verts, faces=template_mesh.faces, process=False)
        save_ply_explicit(stitch_mesh, stitch_file)

        # --- 3) Boundary stitching + Taubin ---
        stitch_taubin_file = os.path.join(output_dir, f'{basename}_completed_stitch_taubin.ply')
        st_verts = raw_verts.copy()
        st_verts, num_affected = postprocess_completion(
            st_verts, faces_np, mask,
            method='stitch+taubin', num_rings=3, iterations=5,
            lambda_factor=0.5, mu_factor=-0.53, stitch_weight=0.7
        )
        st_mesh = Trimesh(vertices=st_verts, faces=template_mesh.faces, process=False)
        save_ply_explicit(st_mesh, stitch_taubin_file)

        # Compute errors for all three
        err_raw = np.linalg.norm(raw_verts - gt_verts, axis=1)
        err_stitch = np.linalg.norm(stitch_verts - gt_verts, axis=1)
        err_st = np.linalg.norm(st_verts - gt_verts, axis=1)

        if verbose:
            r_miss = err_raw[mask].mean() if mask.any() else 0.0
            st_miss = err_stitch[mask].mean() if mask.any() else 0.0
            stt_miss = err_st[mask].mean() if mask.any() else 0.0
            print(f"  {basename}: raw={r_miss:.6f}  stitch={st_miss:.6f}  "
                  f"stitch+taubin={stt_miss:.6f}  (missing region)")

    print(f"\nAll meshes saved to {output_dir}")
    print(f"  *_raw.ply          = model output (no post-processing)")
    print(f"  *_stitch.ply       = boundary stitching only")
    print(f"  *_stitch_taubin.ply = boundary stitching + Taubin smoothing")


def main():
    parser = argparse.ArgumentParser(description='Mesh completion inference')
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to config_completion.cfg")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single partial mesh .ply file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the completed mesh")
    parser.add_argument("--eval_test", action='store_true',
                        help="Evaluate on the saved test samples (partial + GT pairs)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory holding *_gt_complete.ply / *_mask.npy test samples "
                             "(default: <dataset_dir>/completion_test_samples)")
    parser.add_argument("--no_smooth", action='store_true',
                        help="Disable post-processing smoothing")
    args = parser.parse_args()

    config = read_config(args.config_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_completion_model(config, device)
    template_mesh, edge_index = get_template_info(config)
    print(f"Template: {len(template_mesh.vertices)} vertices, {len(template_mesh.faces)} faces")

    smooth = not args.no_smooth

    if args.eval_test:
        evaluate_test_samples(model, config, device, template_mesh, edge_index,
                              smooth=smooth, test_dir=args.test_dir)

    if args.input:
        if args.output is None:
            args.output = args.input.replace('.ply', '_completed.ply')
        complete_single_mesh(model, args.input, args.output, config, device,
                             template_mesh, edge_index, smooth=smooth)


if __name__ == "__main__":
    main()
