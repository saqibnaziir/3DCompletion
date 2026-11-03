"""
CVPR-style qualitative figure for mesh completion.

For each sample, renders a row of 4 panels:
    Ground Truth | Input (partial, hole) | Ours (completed) | Error heatmap

The Input panel shows the true partial mesh (faces touching a missing vertex are
removed, so the hole is visible). The Error panel is a per-vertex distance heatmap
(jet) between Ours and GT, which naturally focuses attention on the filled region.

Outputs:
  - one PNG per sample            (qualitative_results/<sample>.png)
  - one combined grid figure      (qualitative_results/_grid.png)

Usage:
  python scripts/qualitative_figure.py
  python scripts/qualitative_figure.py --samples test_012 test_006 test_004
  python scripts/qualitative_figure.py --view xy --clim 0.012
"""
import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pyvista as pv
from glob import glob
from trimesh import load_mesh

# GT/partial/mask live in TEST_DIR; completed meshes in COMP_DIR (= TEST_DIR\completed_diffusion
# by default, which is where infer_diffusion.py --eval_test --test_dir writes them).
TEST_DIR = os.path.join('data', 'completion_test_samples', 'test_set')
COMP_DIR = os.path.join(TEST_DIR, 'completed_diffusion')
OUT_DIR = os.path.join('data', 'completion_test_samples', 'qualitative_results')
# GT/mask/partial may live in TEST_DIR root (default) or in separate subfolders
# (reorganized layout, e.g. test_set/gt + test_set/partial). Overridable via CLI.
GT_DIR = TEST_DIR
MASK_DIR = TEST_DIR

COL_TITLES = ["GT", "Input", "Ours", "Error"]


def pv_faces(faces):
    return np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64)


def bold_text(pl, text, font_size, position, color='black'):
    """add_text with the text property forced to BOLD (+ shadow for legibility)."""
    actor = pl.add_text(text, font_size=font_size, color=color, position=position)
    try:
        tp = actor.GetTextProperty()
        tp.SetBold(True)
        tp.SetShadow(True)
        tp.SetFontFamilyToArial()
    except Exception:
        pass
    return actor


def add_surface(pl, verts, faces, title, scalars=None, clim=None,
                cmap="jet", show_bar=False, view='xy', zoom=1.1,
                title_color='black', title_font=26, bar_args=None):
    mesh = pv.PolyData(verts, pv_faces(faces))
    if scalars is not None:
        mesh.point_data["err"] = scalars
        # bar_args lets a caller widen/retitle the bar for a wide multi-column
        # plotter, where the default 5%-of-subplot bar renders too small to read.
        bargs = dict(title="", vertical=True,
                     position_x=0.90, position_y=0.25,
                     height=0.5, width=0.05,
                     label_font_size=18)
        bargs.update(bar_args or {})
        pl.add_mesh(mesh, scalars="err", cmap=cmap, clim=clim,
                    show_edges=False, smooth_shading=True,
                    show_scalar_bar=show_bar,
                    scalar_bar_args=bargs)
    else:
        pl.add_mesh(mesh, color="lightgray", show_edges=False,
                    smooth_shading=True, specular=0.3, specular_power=15)
    pl.camera_position = view
    pl.reset_camera()
    pl.camera.zoom(zoom)
    # Title added AFTER camera so it sits in the top band, clear of the mesh.
    if title:
        bold_text(pl, title, font_size=title_font, position='upper_edge',
                  color=title_color)


def load_sample(name):
    gt_path = os.path.join(GT_DIR, f"{name}_gt_complete.ply")
    mask_path = os.path.join(MASK_DIR, f"{name}_mask.npy")
    comp_path = os.path.join(COMP_DIR, f"{name}_completed_diffusion.ply")
    partials = glob(os.path.join(MASK_DIR, f"{name}_partial_*.ply"))
    for p in (gt_path, mask_path, comp_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    gt = load_mesh(gt_path, process=False)
    gt_v = np.array(gt.vertices, dtype=np.float32)
    faces = np.array(gt.faces)
    mask = np.load(mask_path)                        # True = missing
    comp_v = np.array(load_mesh(comp_path, process=False).vertices, dtype=np.float32)

    # Strategy/ratio from the partial filename, for the row label.
    strat = "unknown"
    if partials:
        strat = os.path.basename(partials[0]).replace(f"{name}_partial_", "").replace(".ply", "")

    # Partial mesh = keep only faces whose 3 verts are all known -> visible hole.
    known = ~mask
    keep = known[faces].all(axis=1)
    partial_faces = faces[keep]

    err = np.linalg.norm(comp_v - gt_v, axis=1)
    return dict(name=name, gt_v=gt_v, faces=faces, mask=mask, comp_v=comp_v,
                partial_faces=partial_faces, err=err, strat=strat)


def render_row(pl, row, s, clim, view, zoom, with_titles):
    strat_clean = s['strat'].split('_r')[0]      # drop the redundant ratio suffix
    label = f"{s['name']}   {strat_clean}   {100*s['mask'].mean():.0f}% missing"
    # 0 GT
    pl.subplot(row, 0)
    add_surface(pl, s['gt_v'], s['faces'], COL_TITLES[0] if with_titles else "",
                view=view, zoom=zoom)
    bold_text(pl, label, font_size=18, color='black', position='lower_left')
    # 1 Input (partial w/ hole)
    pl.subplot(row, 1)
    add_surface(pl, s['gt_v'], s['partial_faces'], COL_TITLES[1] if with_titles else "",
                view=view, zoom=zoom)
    # 2 Ours
    pl.subplot(row, 2)
    add_surface(pl, s['comp_v'], s['faces'], COL_TITLES[2] if with_titles else "",
                view=view, zoom=zoom)
    # 3 Error
    pl.subplot(row, 3)
    add_surface(pl, s['comp_v'], s['faces'], COL_TITLES[3] if with_titles else "",
                scalars=s['err'], clim=clim, show_bar=True, view=view, zoom=zoom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs='*', default=None,
                    help="sample ids, e.g. test_012 test_006 (default: a curated set)")
    ap.add_argument("--view", type=str, default='xy', help="pyvista camera_position")
    ap.add_argument("--zoom", type=float, default=1.1)
    ap.add_argument("--clim", type=float, default=None,
                    help="max error for the heatmap colorbar (auto if unset)")
    ap.add_argument("--single", type=str, default=None,
                    help="render ONE sample to a calibration png and exit")
    ap.add_argument("--test_dir", type=str, default=None,
                    help="folder with GT/partial/mask (default: completion_test_samples/test_set)")
    ap.add_argument("--comp_dir", type=str, default=None,
                    help="folder with *_completed_diffusion.ply (default: <test_dir>/completed_diffusion)")
    ap.add_argument("--gt_dir", type=str, default=None,
                    help="folder with *_gt_complete.ply (default: <test_dir>; use for split layout)")
    ap.add_argument("--mask_dir", type=str, default=None,
                    help="folder with *_mask.npy and *_partial_*.ply (default: <test_dir>)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="where PNGs go (default: <comp_dir>/qualitative_results)")
    args = ap.parse_args()

    global TEST_DIR, COMP_DIR, OUT_DIR, GT_DIR, MASK_DIR
    if args.test_dir:
        TEST_DIR = args.test_dir
        COMP_DIR = os.path.join(TEST_DIR, 'completed_diffusion')
    # GT/mask default to TEST_DIR root, overridable for the split layout.
    GT_DIR = args.gt_dir if args.gt_dir else TEST_DIR
    MASK_DIR = args.mask_dir if args.mask_dir else TEST_DIR
    if args.comp_dir:
        COMP_DIR = args.comp_dir
    # Default output sits inside the arm's own folder (matches completed_diffusion_*/
    # qualitative_results), so different arms never overwrite each other's figures.
    if args.out_dir:
        OUT_DIR = args.out_dir
    else:
        OUT_DIR = os.path.join(COMP_DIR, 'qualitative_results')
    print(f"GT dir  : {GT_DIR}\nMask dir: {MASK_DIR}\nCompleted dir : {COMP_DIR}\nOutput dir: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.single:
        s = load_sample(args.single)
        clim = [0, args.clim or float(np.percentile(s['err'], 99))]
        pl = pv.Plotter(off_screen=True, shape=(1, 4), window_size=[2000, 560], border=False)
        pl.background_color = 'white'
        render_row(pl, 0, s, clim, args.view, args.zoom, with_titles=True)
        out = os.path.join(OUT_DIR, f"_calib_{args.single}.png")
        pl.screenshot(out); pl.close()
        print("wrote", out, " err p99=", clim[1])
        return

    # Curated default: contiguous holes read best (bfs / plane), plus one scatter.
    samples = args.samples or ["test_012", "test_006", "test_013", "test_004",
                               "test_010", "test_001"]
    data = [load_sample(n) for n in samples]

    # Shared color scale across all rows for fair comparison.
    if args.clim is not None:
        vmax = args.clim
    else:
        vmax = float(np.percentile(np.concatenate([d['err'] for d in data]), 99))
    clim = [0.0, vmax]
    print(f"Error colorbar range: [0, {vmax:.5f}]")

    # Per-sample PNGs.
    for s in data:
        pl = pv.Plotter(off_screen=True, shape=(1, 4), window_size=[2000, 560], border=False)
        pl.background_color = 'white'
        render_row(pl, 0, s, clim, args.view, args.zoom, with_titles=True)
        out = os.path.join(OUT_DIR, f"{s['name']}.png")
        pl.screenshot(out); pl.close()
        print("wrote", out)

    # Combined grid.
    R = len(data)
    pl = pv.Plotter(off_screen=True, shape=(R, 4), window_size=[2000, 520 * R], border=False)
    pl.background_color = 'white'
    for r, s in enumerate(data):
        render_row(pl, r, s, clim, args.view, args.zoom, with_titles=(r == 0))
    grid = os.path.join(OUT_DIR, "_grid.png")
    pl.screenshot(grid); pl.close()
    print("wrote", grid)


if __name__ == "__main__":
    main()
