"""
Prepare the DFAUST (Dynamic FAUST) registrations for the mesh-completion pipeline.

The registrations_{m,f}.hdf5 files hold SMPL-topology bodies (6890 verts, 13776
faces) as (6890, 3, num_frames) per subject_sequence. This script subsamples the
frames (they are 60 fps and highly redundant) and writes each frame as a binary
PLY into <dataset_dir>/train/<subject_seq>/<frame>.ply -- exactly the layout the
existing MeshCompletionDataset.process() globs for (mirrors COMA's data/train).

After this runs, the rest of the pipeline (dataset build, diffusion training,
inference) is identical to COMA and driven by research/configs/config_diffusion_dfaust.cfg.

Run:
  python research/prepare_dfaust.py --stride 6
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import numpy as np
import h5py

from config.config import resolve_config_file, read_config


def write_binary_ply(path, verts, faces):
    """Minimal binary_little_endian PLY writer (no trimesh dependency)."""
    verts = np.ascontiguousarray(verts, dtype='<f4')
    faces = np.ascontiguousarray(faces, dtype='<i4')
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode('ascii')
    # each face row: uchar count (3) + 3 int32 -> vectorized structured array
    face_dtype = np.dtype([('c', 'u1'), ('v', '<i4', 3)])
    face_rec = np.empty(len(faces), dtype=face_dtype)
    face_rec['c'] = 3
    face_rec['v'] = faces
    with open(path, 'wb') as fp:
        fp.write(header)
        fp.write(verts.tobytes())
        fp.write(face_rec.tobytes())


def export_hdf5(h5_path, out_dir, stride, faces_out):
    n_written = 0
    with h5py.File(h5_path, 'r') as f:
        faces = f['faces'][()].astype(np.int32)          # (13776, 3)
        if faces_out[0] is None:
            faces_out[0] = faces
        seqs = [k for k in f.keys() if k != 'faces']
        for seq in sorted(seqs):
            arr = f[seq][()]                              # (6890, 3, T)
            T = arr.shape[-1]
            seq_dir = os.path.join(out_dir, seq)
            os.makedirs(seq_dir, exist_ok=True)
            for t in range(0, T, stride):
                verts = arr[:, :, t]                      # (6890, 3)
                out_path = os.path.join(seq_dir, f'{t:05d}.ply')
                write_binary_ply(out_path, verts, faces)
                n_written += 1
            print(f"  {seq}: {len(range(0, T, stride))} frames (of {T})")
    return n_written


def main():
    ap = argparse.ArgumentParser(description="Export DFAUST registrations to PLY")
    ap.add_argument('--data', type=str, default='DFAUST')
    ap.add_argument('--config_file', type=str, default=None)
    ap.add_argument('--dfaust_dir', type=str,
                    default=os.path.join('data', 'DEFAUST'),
                    help='folder holding registrations_{m,f}.hdf5')
    ap.add_argument('--stride', type=int, default=6,
                    help='keep every Nth frame (60 fps source is very redundant)')
    args = ap.parse_args()

    cfg = read_config(resolve_config_file(args.data, args.config_file))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dfaust_dir = args.dfaust_dir
    if not os.path.isabs(dfaust_dir):
        dfaust_dir = os.path.join(project_root, dfaust_dir)

    out_dir = os.path.join(cfg['dataset_dir'], 'train')
    os.makedirs(out_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(dfaust_dir, 'registrations_*.hdf5')))
    if not h5_files:
        raise FileNotFoundError(f"No registrations_*.hdf5 found in {dfaust_dir}")

    print(f"DFAUST source : {dfaust_dir}")
    print(f"Output train  : {out_dir}")
    print(f"Frame stride  : {args.stride}")
    print(f"HDF5 files    : {[os.path.basename(x) for x in h5_files]}")

    faces_out = [None]
    total = 0
    for h5 in h5_files:
        print(f"\n== {os.path.basename(h5)} ==")
        total += export_hdf5(h5, out_dir, args.stride, faces_out)

    print(f"\nDone. Wrote {total} meshes to {out_dir}")
    print(f"Vertices/mesh: {faces_out[0].max() + 1} | faces/mesh: {len(faces_out[0])}")
    print("Next: build the dataset + train (see runCMD.md 'DFAUST' section).")


if __name__ == '__main__':
    main()
