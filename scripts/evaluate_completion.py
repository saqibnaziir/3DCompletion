"""
Comprehensive evaluation of mesh completion results.

For each test sample, produces an 8-panel comparison:
  Row 1: GT | Partial | Raw | Stitched | Stitch+Taubin
  Row 2: Error (Raw) | Error (Stitched) | Error (Stitch+Taubin)

Plus quantitative metrics for all three variants in a CSV.

Usage:
  python scripts/evaluate_completion.py --config_file config/config_completion.cfg
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
import csv
from glob import glob
from trimesh import load_mesh


def render_mesh(plotter, mesh_verts, mesh_faces, row, col, title,
                scalars=None, cmap="jet", clim=None, show_scalar_bar=False):
    """Add a mesh to a specific subplot position."""
    plotter.subplot(row, col)
    pv_faces = np.column_stack((np.full(len(mesh_faces), 3), mesh_faces))
    pv_mesh = pv.PolyData(mesh_verts, pv_faces)

    if scalars is not None:
        pv_mesh.point_data["values"] = scalars
        plotter.add_mesh(pv_mesh, scalars="values", cmap=cmap,
                         show_edges=False, clim=clim,
                         show_scalar_bar=show_scalar_bar)
    else:
        plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False)

    plotter.add_text(title, font_size=10, position='upper_edge')
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.camera.zoom(1.3)


def visualize_sample(gt_verts, partial_verts, raw_verts, stitch_verts, st_verts,
                     mask, faces, errors_raw, errors_stitch, errors_st,
                     sample_name, output_dir):
    """
    8-panel figure (2 rows x 4 columns):
      Row 1: GT | Partial (red=missing) | Raw | Stitched | Stitch+Taubin
      Row 2: (empty) | Error (Raw) | Error (Stitched) | Error (Stitch+Taubin)
    
    Laid out as 2x4 grid for clarity.
    """
    pv_faces = np.column_stack((np.full(len(faces), 3), faces))

    # Shared error clim across all three for fair comparison
    max_err = max(
        np.percentile(errors_raw, 95),
        np.percentile(errors_stitch, 95),
        np.percentile(errors_st, 95)
    )
    error_clim = [0, max_err]

    plotter = pv.Plotter(off_screen=True, shape=(2, 4),
                         window_size=[4000, 2000], border=False)
    plotter.background_color = 'white'

    # --- Row 0: Mesh views ---

    # (0,0) GT
    render_mesh(plotter, gt_verts, faces, 0, 0, "Ground Truth")

    # (0,1) Partial (known=gray, missing=red)
    plotter.subplot(0, 1)
    pv_mesh_p = pv.PolyData(gt_verts, pv_faces)
    mask_float = mask.astype(np.float32)
    pv_mesh_p.point_data["mask"] = mask_float
    plotter.add_mesh(pv_mesh_p, scalars="mask", cmap=["lightgray", "red"],
                     show_edges=False, clim=[0, 1], show_scalar_bar=False)
    plotter.add_text("Partial (red=missing)", font_size=10, position='upper_edge')
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.camera.zoom(1.3)

    # (0,2) Stitched
    render_mesh(plotter, stitch_verts, faces, 0, 2, "Stitched")

    # (0,3) Stitch + Taubin
    render_mesh(plotter, st_verts, faces, 0, 3, "Stitch + Taubin")

    # --- Row 1: Error maps ---

    # (1,0) Raw completed mesh
    render_mesh(plotter, raw_verts, faces, 1, 0, "Raw Completed")

    # (1,1) Error map — Raw
    render_mesh(plotter, raw_verts, faces, 1, 1, "Error (Raw)",
                scalars=errors_raw, cmap="jet", clim=error_clim,
                show_scalar_bar=True)

    # (1,2) Error map — Stitched
    render_mesh(plotter, stitch_verts, faces, 1, 2, "Error (Stitched)",
                scalars=errors_stitch, cmap="jet", clim=error_clim,
                show_scalar_bar=True)

    # (1,3) Error map — Stitch+Taubin
    render_mesh(plotter, st_verts, faces, 1, 3, "Error (Stitch+Taubin)",
                scalars=errors_st, cmap="jet", clim=error_clim,
                show_scalar_bar=True)

    comparison_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
    plotter.screenshot(comparison_path, transparent_background=False)
    plotter.close()
    print(f"  Saved: {comparison_path}")


def compute_metrics(gt_verts, completed_verts, mask, prefix=''):
    """Compute quantitative error metrics with an optional key prefix."""
    per_vertex_error = np.linalg.norm(gt_verts - completed_verts, axis=1)

    p = f"{prefix}_" if prefix else ""
    metrics = {
        f'{p}overall_mean': per_vertex_error.mean(),
        f'{p}overall_median': np.median(per_vertex_error),
        f'{p}overall_max': per_vertex_error.max(),
        f'{p}overall_std': per_vertex_error.std(),
    }

    if mask.any():
        m = per_vertex_error[mask]
        metrics[f'{p}missing_mean'] = m.mean()
        metrics[f'{p}missing_median'] = np.median(m)
        metrics[f'{p}missing_max'] = m.max()
        metrics[f'{p}missing_std'] = m.std()
    else:
        metrics[f'{p}missing_mean'] = 0.0
        metrics[f'{p}missing_median'] = 0.0
        metrics[f'{p}missing_max'] = 0.0
        metrics[f'{p}missing_std'] = 0.0

    if (~mask).any():
        k = per_vertex_error[~mask]
        metrics[f'{p}known_mean'] = k.mean()
    else:
        metrics[f'{p}known_mean'] = 0.0

    return per_vertex_error, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate mesh completion results')
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    if args.test_dir:
        test_dir = args.test_dir
    elif args.config_file:
        from config.config import read_config
        config = read_config(args.config_file)
        test_dir = os.path.join(config['dataset_dir'], 'completion_test_samples')
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        test_dir = os.path.join(project_root, 'data', 'completion_test_samples')

    completed_dir = os.path.join(test_dir, 'completed')
    output_dir = os.path.join(test_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Test samples dir: {test_dir}")
    print(f"Completed meshes: {completed_dir}")
    print(f"Output dir:       {output_dir}\n")

    gt_files = sorted(glob(os.path.join(test_dir, '*_gt_complete.ply')))
    if args.num_samples:
        gt_files = gt_files[:args.num_samples]

    print(f"Processing {len(gt_files)} test samples...\n")

    all_metrics = []

    for gt_file in gt_files:
        basename = os.path.basename(gt_file).replace('_gt_complete.ply', '')

        mask_file = os.path.join(test_dir, f'{basename}_mask.npy')
        raw_file = os.path.join(completed_dir, f'{basename}_completed_raw.ply')
        stitch_file = os.path.join(completed_dir, f'{basename}_completed_stitch.ply')
        st_file = os.path.join(completed_dir, f'{basename}_completed_stitch_taubin.ply')

        if not os.path.exists(mask_file):
            print(f"  Skipping {basename}: no mask file")
            continue
        if not os.path.exists(raw_file):
            print(f"  Skipping {basename}: no raw completed file")
            continue
        if not os.path.exists(stitch_file):
            print(f"  Skipping {basename}: no stitched file")
            continue
        if not os.path.exists(st_file):
            print(f"  Skipping {basename}: no stitch+taubin file")
            continue

        # Load everything
        gt_mesh = load_mesh(gt_file, process=False)
        gt_verts = np.array(gt_mesh.vertices, dtype=np.float32)
        faces = np.array(gt_mesh.faces)
        mask = np.load(mask_file)

        raw_mesh = load_mesh(raw_file, process=False)
        raw_verts = np.array(raw_mesh.vertices, dtype=np.float32)

        stitch_mesh = load_mesh(stitch_file, process=False)
        stitch_verts = np.array(stitch_mesh.vertices, dtype=np.float32)

        st_mesh = load_mesh(st_file, process=False)
        st_verts = np.array(st_mesh.vertices, dtype=np.float32)

        # Validate shapes
        if (gt_verts.shape != raw_verts.shape or
            gt_verts.shape != stitch_verts.shape or
            gt_verts.shape != st_verts.shape):
            print(f"  Skipping {basename}: shape mismatch")
            continue

        # Build partial for visualization (from GT + mask)
        partial_verts = gt_verts.copy()
        partial_verts[mask] = 0.0

        # Compute errors for all three methods
        errors_raw, metrics_raw = compute_metrics(gt_verts, raw_verts, mask, prefix='raw')
        errors_stitch, metrics_stitch = compute_metrics(gt_verts, stitch_verts, mask, prefix='stitch')
        errors_st, metrics_st = compute_metrics(gt_verts, st_verts, mask, prefix='stitch_taubin')

        # Masking info from filename
        partial_files = glob(os.path.join(test_dir, f'{basename}_partial_*.ply'))
        masking_info = 'unknown'
        if partial_files:
            pname = os.path.basename(partial_files[0])
            masking_info = pname.replace(f'{basename}_partial_', '').replace('.ply', '')

        sample_metrics = {
            'sample': basename,
            'masking_info': masking_info,
            'missing_count': int(mask.sum()),
            'missing_pct': 100.0 * mask.sum() / len(mask),
        }
        sample_metrics.update(metrics_raw)
        sample_metrics.update(metrics_stitch)
        sample_metrics.update(metrics_st)
        all_metrics.append(sample_metrics)

        # Print comparison
        print(f"  {basename} ({masking_info}):")
        print(f"    Missing: {sample_metrics['missing_count']} ({sample_metrics['missing_pct']:.1f}%)")
        print(f"    RAW           — missing: {metrics_raw['raw_missing_mean']:.6f}  "
              f"overall: {metrics_raw['raw_overall_mean']:.6f}  "
              f"known: {metrics_raw['raw_known_mean']:.6f}")
        print(f"    STITCH        — missing: {metrics_stitch['stitch_missing_mean']:.6f}  "
              f"overall: {metrics_stitch['stitch_overall_mean']:.6f}  "
              f"known: {metrics_stitch['stitch_known_mean']:.6f}")
        print(f"    STITCH+TAUBIN — missing: {metrics_st['stitch_taubin_missing_mean']:.6f}  "
              f"overall: {metrics_st['stitch_taubin_overall_mean']:.6f}  "
              f"known: {metrics_st['stitch_taubin_known_mean']:.6f}")

        # Visualize (8-panel: 2x4 grid)
        try:
            visualize_sample(gt_verts, partial_verts, raw_verts,
                             stitch_verts, st_verts,
                             mask, faces,
                             errors_raw, errors_stitch, errors_st,
                             basename, output_dir)
        except Exception as e:
            print(f"    WARNING: Visualization failed: {e}")

    # --- Aggregate ---
    if all_metrics:
        print("\n" + "=" * 80)
        print(f"AGGREGATE RESULTS over {len(all_metrics)} samples")
        print("=" * 80)

        miss_pcts = [m['missing_pct'] for m in all_metrics]

        raw_miss = [m['raw_missing_mean'] for m in all_metrics]
        raw_all = [m['raw_overall_mean'] for m in all_metrics]
        raw_known = [m['raw_known_mean'] for m in all_metrics]

        stitch_miss = [m['stitch_missing_mean'] for m in all_metrics]
        stitch_all = [m['stitch_overall_mean'] for m in all_metrics]
        stitch_known = [m['stitch_known_mean'] for m in all_metrics]

        st_miss = [m['stitch_taubin_missing_mean'] for m in all_metrics]
        st_all = [m['stitch_taubin_overall_mean'] for m in all_metrics]
        st_known = [m['stitch_taubin_known_mean'] for m in all_metrics]

        print(f"  Avg missing %:              {np.mean(miss_pcts):.1f}%")
        print(f"")
        print(f"  RAW           missing error: {np.mean(raw_miss):.6f} +/- {np.std(raw_miss):.6f}")
        print(f"  STITCH        missing error: {np.mean(stitch_miss):.6f} +/- {np.std(stitch_miss):.6f}")
        print(f"  STITCH+TAUBIN missing error: {np.mean(st_miss):.6f} +/- {np.std(st_miss):.6f}")
        print(f"")
        print(f"  RAW           overall error: {np.mean(raw_all):.6f} +/- {np.std(raw_all):.6f}")
        print(f"  STITCH        overall error: {np.mean(stitch_all):.6f} +/- {np.std(stitch_all):.6f}")
        print(f"  STITCH+TAUBIN overall error: {np.mean(st_all):.6f} +/- {np.std(st_all):.6f}")
        print(f"")
        print(f"  RAW           known error:   {np.mean(raw_known):.6f}")
        print(f"  STITCH        known error:   {np.mean(stitch_known):.6f}")
        print(f"  STITCH+TAUBIN known error:   {np.mean(st_known):.6f}")

        # Save CSV
        csv_path = os.path.join(output_dir, 'completion_metrics.csv')
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)
        print(f"\n  Metrics CSV: {csv_path}")

    print(f"  Visualizations: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
