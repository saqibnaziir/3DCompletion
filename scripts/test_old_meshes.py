"""
Test the trained completion model on the old test meshes (GT + partial pairs).

Usage:
  python scripts/test_old_meshes.py --config_file config/config_completion.cfg

Reads from:  D:\COURSAI\3Dis\data\test_old\gt\   and   D:\COURSAI\3Dis\data\test_old\partial\
Saves to:    D:\COURSAI\3Dis\data\test_old\completed\  and  D:\COURSAI\3Dis\data\test_old\evaluation\
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import csv
from scipy.spatial import cKDTree
from trimesh import Trimesh, load_mesh

from config.config import read_config
from utils.funcs import get_mesh_matrices, save_ply_explicit, get_edge_index
from utils.models import FMGenModel
from utils.postprocess import postprocess_completion

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("WARNING: pyvista not installed, skipping visualization")


# ---- Pairs mapping: GT filename -> Partial filename ----
PAIRS = [
    ("gt_ear.ply",   "no_ear.ply"),
    ("gt_eyes.ply",  "no_eyes.ply"),
    ("gt_half.ply",  "no_half.ply"),
    ("gt_half2.ply", "no_half2.ply"),
    ("gt_mouth.ply", "no_mouth.ply"),
    ("gt_nose.ply",  "no_nose.ply"),
    ("gt_plain.ply", "no_plane.ply"),
]


def reconstruct_fixed_topology(gt_verts, partial_verts, distance_threshold=0.01):
    """
    Reconstruct a fixed-topology mesh from a partial mesh with removed vertices.

    The partial meshes in test_old have actually DELETED vertices (different count),
    rather than zeroing them. We need to figure out which GT vertices are present
    in the partial mesh and which are missing.

    Strategy:
      1. Build a KDTree from partial mesh vertices
      2. For each GT vertex, find its nearest neighbor in the partial mesh
      3. If distance < threshold → vertex is "known" (present in partial)
      4. If distance >= threshold → vertex is "missing" (was removed)
      5. Return a full (N_template, 3) array with missing positions zeroed

    Args:
        gt_verts: (N, 3) complete GT vertices
        partial_verts: (M, 3) partial mesh vertices (M < N)
        distance_threshold: max distance to consider a match

    Returns:
        fixed_verts: (N, 3) with missing vertices zeroed
        missing_mask: (N,) bool, True = missing
    """
    N = len(gt_verts)
    tree = cKDTree(partial_verts)
    distances, indices = tree.query(gt_verts)

    # A GT vertex is "known" if its nearest partial vertex is very close
    missing_mask = distances > distance_threshold
    
    # Build the fixed-topology mesh: known vertices keep GT coords, missing are zeroed
    fixed_verts = gt_verts.copy()
    fixed_verts[missing_mask] = 0.0

    return fixed_verts, missing_mask


def load_model(config, device):
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

    print(f"Loaded model from {checkpoint_dir}")
    return model


def load_norm(config):
    """Load normalization parameters."""
    norm_path = os.path.join(config['dataset_dir'], 'norm_completion.pt')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(config['dataset_dir'], 'norm.pt')
    norm_dict = torch.load(norm_path, map_location='cpu', weights_only=False)
    mean = norm_dict['mean'].numpy()
    std = norm_dict['std']
    std_np = std.numpy() if std is not None else None
    mean_subtraction_only = std is None
    return mean, std_np, mean_subtraction_only


def complete_mesh(model, partial_verts, missing_mask, mean, std, mean_sub_only,
                  edge_index, template_faces, device):
    """Run a single completion forward pass and return raw + post-processed results."""
    from torch_geometric.data import Data, Batch

    N = len(partial_verts)

    # Normalize
    verts_norm = partial_verts.astype(np.float32).copy() - mean
    if not mean_sub_only and std is not None:
        verts_norm = verts_norm / std
    verts_norm[missing_mask] = 0.0

    # Build 4-channel input
    mask_ch = np.ones((N, 1), dtype=np.float32)
    mask_ch[missing_mask] = 0.0
    input_feat = np.concatenate([verts_norm, mask_ch], axis=1)

    input_tensor = torch.tensor(input_feat, dtype=torch.float32).to(device)
    data = Data(x=input_tensor, edge_index=edge_index.to(device))
    batch = Batch.from_data_list([data])

    with torch.no_grad():
        out, z = model(batch)

    # Denormalize
    out_np = out.cpu().numpy()
    if not mean_sub_only and std is not None:
        out_np = out_np * std
    out_np = out_np + mean

    # Raw: blend known + predicted
    raw_verts = partial_verts.copy()
    raw_verts[missing_mask] = out_np[missing_mask]

    # Stitch only
    faces_np = np.array(template_faces)
    stitch_verts, _ = postprocess_completion(
        raw_verts.copy(), faces_np, missing_mask,
        method='stitch', num_rings=3, stitch_weight=0.7
    )

    # Stitch + Taubin
    st_verts, _ = postprocess_completion(
        raw_verts.copy(), faces_np, missing_mask,
        method='stitch+taubin', num_rings=3, iterations=5,
        lambda_factor=0.5, mu_factor=-0.53, stitch_weight=0.7
    )

    return raw_verts, stitch_verts, st_verts


def render_comparison(gt_verts, partial_verts, raw_verts, stitch_verts, st_verts,
                      missing_mask, faces, errors_raw, errors_stitch, errors_st,
                      name, output_dir):
    """Render 2x4 comparison panel."""
    if not HAS_PYVISTA:
        return

    pv_faces = np.column_stack((np.full(len(faces), 3), faces))
    max_err = max(
        np.percentile(errors_raw, 95),
        np.percentile(errors_stitch, 95),
        np.percentile(errors_st, 95),
        1e-8
    )
    clim = [0, max_err]

    plotter = pv.Plotter(off_screen=True, shape=(2, 4), window_size=[4000, 2000], border=False)
    plotter.background_color = 'white'

    def add_mesh(row, col, verts, title, scalars=None):
        plotter.subplot(row, col)
        mesh = pv.PolyData(verts, pv_faces)
        if scalars is not None:
            mesh.point_data["v"] = scalars
            plotter.add_mesh(mesh, scalars="v", cmap="jet", clim=clim,
                             show_edges=False, show_scalar_bar=(col >= 1 and row == 1))
        else:
            plotter.add_mesh(mesh, color="lightgray", show_edges=False)
        plotter.add_text(title, font_size=10, position='upper_edge')
        plotter.camera_position = 'xy'
        plotter.reset_camera()
        plotter.camera.zoom(1.3)

    # Row 0: meshes
    add_mesh(0, 0, gt_verts, "Ground Truth")

    # Partial with red highlighting
    plotter.subplot(0, 1)
    pmesh = pv.PolyData(gt_verts, pv_faces)
    pmesh.point_data["mask"] = missing_mask.astype(np.float32)
    plotter.add_mesh(pmesh, scalars="mask", cmap=["lightgray", "red"],
                     clim=[0, 1], show_edges=False, show_scalar_bar=False)
    plotter.add_text(f"Partial ({name})", font_size=10, position='upper_edge')
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.camera.zoom(1.3)

    add_mesh(0, 2, stitch_verts, "Stitched")
    add_mesh(0, 3, st_verts, "Stitch+Taubin")

    # Row 1: raw + error maps
    add_mesh(1, 0, raw_verts, "Raw Completed")
    add_mesh(1, 1, raw_verts, "Error (Raw)", scalars=errors_raw)
    add_mesh(1, 2, stitch_verts, "Error (Stitch)", scalars=errors_stitch)
    add_mesh(1, 3, st_verts, "Error (Stitch+Tau)", scalars=errors_st)

    out_path = os.path.join(output_dir, f"{name}_comparison.png")
    plotter.screenshot(out_path, transparent_background=False)
    plotter.close()
    print(f"    Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Test completion on old test meshes')
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=r"D:\COURSAI\3Dis\data\test_old")
    args = parser.parse_args()

    config = read_config(args.config_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model, normalization, template
    model = load_model(config, device)
    mean, std, mean_sub_only = load_norm(config)
    template = load_mesh(config['template_file'], process=False)
    N_template = len(template.vertices)
    print(f"Template: {N_template} vertices, {len(template.faces)} faces")

    # Edge index
    ei_path = os.path.join(config['dataset_dir'], 'processed_completion', 'edge_index.pt')
    if os.path.exists(ei_path):
        edge_index = torch.load(ei_path, weights_only=False)
    else:
        edge_index = get_edge_index(template.vertices, template.faces)

    # Directories
    test_dir = args.test_dir
    gt_dir = os.path.join(test_dir, "gt")
    partial_dir = os.path.join(test_dir, "partial")
    completed_dir = os.path.join(test_dir, "completed")
    eval_dir = os.path.join(test_dir, "evaluation")
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\nGT dir:        {gt_dir}")
    print(f"Partial dir:   {partial_dir}")
    print(f"Output:        {completed_dir}")
    print(f"Evaluation:    {eval_dir}")

    all_metrics = []

    for gt_file, partial_file in PAIRS:
        gt_path = os.path.join(gt_dir, gt_file)
        partial_path = os.path.join(partial_dir, partial_file)

        if not os.path.exists(gt_path) or not os.path.exists(partial_path):
            print(f"\n  Skipping {gt_file} / {partial_file}: file not found")
            continue

        name = gt_file.replace("gt_", "").replace(".ply", "")
        print(f"\n{'='*60}")
        print(f"  Processing: {name}")
        print(f"    GT: {gt_file}  |  Partial: {partial_file}")

        # Load meshes (process=False to preserve vertex order)
        gt_mesh = load_mesh(gt_path, process=False)
        partial_mesh = load_mesh(partial_path, process=False)

        gt_verts = np.array(gt_mesh.vertices, dtype=np.float32)
        partial_verts_raw = np.array(partial_mesh.vertices, dtype=np.float32)

        print(f"    GT verts: {len(gt_verts)}  |  Partial verts: {len(partial_verts_raw)}  |  Template: {N_template}")

        # Check GT compatibility
        if len(gt_verts) != N_template:
            print(f"    WARNING: GT vertex count ({len(gt_verts)}) != template ({N_template}). Skipping.")
            continue

        # Handle partial mesh with different vertex count
        if len(partial_verts_raw) == N_template:
            # Same topology — detect missing by zeroed coordinates
            partial_verts = partial_verts_raw
            missing_mask = np.linalg.norm(partial_verts, axis=1) < 1e-6
            print(f"    Same topology — detecting zeroed vertices")
        else:
            # Different vertex count — vertices were actually deleted
            # Reconstruct fixed topology by matching partial → GT via nearest neighbor
            print(f"    Different topology — reconstructing via nearest-neighbor matching...")
            partial_verts, missing_mask = reconstruct_fixed_topology(
                gt_verts, partial_verts_raw, distance_threshold=0.01
            )
            print(f"    Matched {(~missing_mask).sum()} of {N_template} GT vertices to partial mesh")

        num_missing = missing_mask.sum()
        pct_missing = 100.0 * num_missing / N_template
        print(f"    Missing: {num_missing} vertices ({pct_missing:.1f}%)")

        if num_missing == 0:
            print(f"    No missing vertices detected, skipping completion.")
            continue

        # Run completion
        raw_verts, stitch_verts, st_verts = complete_mesh(
            model, partial_verts, missing_mask, mean, std, mean_sub_only,
            edge_index, template.faces, device
        )

        # Save completed meshes
        for suffix, verts in [("raw", raw_verts), ("stitch", stitch_verts), ("stitch_taubin", st_verts)]:
            out_path = os.path.join(completed_dir, f"{name}_completed_{suffix}.ply")
            m = Trimesh(vertices=verts, faces=template.faces, process=False)
            save_ply_explicit(m, out_path)
        print(f"    Saved 3 completed meshes to {completed_dir}")

        # Compute errors
        err_raw = np.linalg.norm(raw_verts - gt_verts, axis=1)
        err_stitch = np.linalg.norm(stitch_verts - gt_verts, axis=1)
        err_st = np.linalg.norm(st_verts - gt_verts, axis=1)

        metrics = {
            'name': name,
            'missing_count': int(num_missing),
            'missing_pct': pct_missing,
            'raw_missing_mean': err_raw[missing_mask].mean(),
            'raw_overall_mean': err_raw.mean(),
            'raw_known_mean': err_raw[~missing_mask].mean(),
            'raw_missing_max': err_raw[missing_mask].max(),
            'stitch_missing_mean': err_stitch[missing_mask].mean(),
            'stitch_overall_mean': err_stitch.mean(),
            'stitch_known_mean': err_stitch[~missing_mask].mean(),
            'stitch_taubin_missing_mean': err_st[missing_mask].mean(),
            'stitch_taubin_overall_mean': err_st.mean(),
            'stitch_taubin_known_mean': err_st[~missing_mask].mean(),
        }
        all_metrics.append(metrics)

        print(f"    RAW           — missing: {metrics['raw_missing_mean']:.6f}  "
              f"overall: {metrics['raw_overall_mean']:.6f}  max: {metrics['raw_missing_max']:.6f}")
        print(f"    STITCH        — missing: {metrics['stitch_missing_mean']:.6f}  "
              f"overall: {metrics['stitch_overall_mean']:.6f}")
        print(f"    STITCH+TAUBIN — missing: {metrics['stitch_taubin_missing_mean']:.6f}  "
              f"overall: {metrics['stitch_taubin_overall_mean']:.6f}")

        # Visualize
        try:
            render_comparison(
                gt_verts, partial_verts, raw_verts, stitch_verts, st_verts,
                missing_mask, np.array(template.faces),
                err_raw, err_stitch, err_st,
                name, eval_dir
            )
        except Exception as e:
            print(f"    Visualization failed: {e}")

    # ---- Aggregate ----
    if all_metrics:
        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS over {len(all_metrics)} test meshes")
        print(f"{'='*70}")

        for key_prefix, label in [('raw', 'RAW'), ('stitch', 'STITCH'), ('stitch_taubin', 'STITCH+TAUBIN')]:
            miss_vals = [m[f'{key_prefix}_missing_mean'] for m in all_metrics]
            overall_vals = [m[f'{key_prefix}_overall_mean'] for m in all_metrics]
            known_vals = [m[f'{key_prefix}_known_mean'] for m in all_metrics]
            print(f"  {label:<15} — missing: {np.mean(miss_vals):.6f} +/- {np.std(miss_vals):.6f}  "
                  f"overall: {np.mean(overall_vals):.6f}  known: {np.mean(known_vals):.6f}")

        pcts = [m['missing_pct'] for m in all_metrics]
        print(f"  Avg missing: {np.mean(pcts):.1f}%  (range: {min(pcts):.1f}% - {max(pcts):.1f}%)")

        # Save CSV
        csv_path = os.path.join(eval_dir, "old_test_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)
        print(f"\n  Metrics CSV: {csv_path}")

    print(f"\nDone. Results in {test_dir}")


if __name__ == "__main__":
    main()
