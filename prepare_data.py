"""
Step 1 of 3: prepare the data.

    python prepare_data.py [--config config/coma.cfg | config/dfaust.cfg]

Reads the raw meshes from `data_dir` and writes into `work_dir`:
    splits/split.json, train.txt, test_<cell>.txt   the split
    data/                                           processed cache, template, normalisation
    benchmark/<cell>/masks.npz, index.csv           fixed test holes, one mask per strategy

Accepted inputs in data_dir:
    <subject>/<sequence>/<name>.ply        COMA layout (sequence = expression)
    <subject>_<sequence>/<frame>.ply       DFAUST exported layout (sequence = motion)
    registrations_*.hdf5                   DFAUST download; exported to <work_dir>/meshes first

Test cells (empty ones are skipped):
    seen_subject       training subjects, held-out frames (COMA: the official test frames)
    unseen_subject     test subjects, training sequences
    unseen_expression  training subjects, test sequences
    unseen_both        test subjects, test sequences

Leakage checks run before and after processing; the script stops on any failure.
Re-running is safe: finished steps are kept. To change the split, delete work_dir.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # same as train_diffusion.py
import sys
import csv
import json
import random
import hashlib
import argparse
import configparser
from glob import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from trimesh import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config.config import read_config
from utils.masking import MASK_STRATEGIES, _build_adjacency

CELLS = ('seen_subject', 'unseen_subject', 'unseen_expression', 'unseen_both')


def abspath(p):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def csv_list(s):
    return [x.strip() for x in s.split(',') if x.strip()]


def write_binary_ply(path, verts, faces):
    verts = np.ascontiguousarray(verts, dtype='<f4')
    rec = np.empty(len(faces), dtype=np.dtype([('c', 'u1'), ('v', '<i4', 3)]))
    rec['c'], rec['v'] = 3, faces
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n'
              f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    with open(path, 'wb') as fh:
        fh.write(header.encode('ascii'))
        fh.write(verts.tobytes())
        fh.write(rec.tobytes())


def export_hdf5(h5_files, out_dir, stride):
    """DFAUST registrations_{m,f}.hdf5 -> <out_dir>/<subject>/<motion>/<frame>.ply."""
    import h5py
    done = os.path.join(out_dir, '.export_done')
    if os.path.exists(done):
        return
    print(f'exporting {len(h5_files)} HDF5 file(s) to {out_dir} (every {stride}th frame) ...')
    for h5 in h5_files:
        with h5py.File(h5, 'r') as f:
            faces = f['faces'][()].astype(np.int32)
            for seq in sorted(k for k in f.keys() if k != 'faces'):
                subject, motion = seq.split('_', 1)
                arr = f[seq][()]                              # (V, 3, T)
                seq_dir = os.path.join(out_dir, subject, motion)
                os.makedirs(seq_dir, exist_ok=True)
                for t in range(0, arr.shape[-1], stride):
                    write_binary_ply(os.path.join(seq_dir, f'{t:05d}.ply'), arr[:, :, t], faces)
    open(done, 'w').close()


def scan(dirs):
    """{(subject, sequence, filename): path} for both accepted folder layouts."""
    meshes = {}
    for d in dirs:
        for f in sorted(glob(os.path.join(d, '*', '*', '*.ply'))):
            p = os.path.normpath(f).split(os.sep)
            meshes.setdefault((p[-3], p[-2], p[-1]), f)
        for f in sorted(glob(os.path.join(d, '*', '*.ply'))):
            p = os.path.normpath(f).split(os.sep)
            if '_' in p[-2]:
                subject, seq = p[-2].split('_', 1)
                meshes.setdefault((subject, seq, p[-1]), f)
    return meshes


def frame_of(key):
    # 'bareteeth.000123.ply' (COMA) or '00123.ply' (DFAUST)
    return int(os.path.splitext(key[2])[0].split('.')[-1])


def make_split(meshes, sp):
    subjects = sorted({k[0] for k in meshes})
    expressions = sorted({k[1] for k in meshes})
    s_test = set(csv_list(sp.get('test_subjects', '')))
    e_test = set(csv_list(sp.get('test_expressions', '')))
    for name, given, known in (('subject', s_test, subjects), ('sequence', e_test, expressions)):
        unknown = given - set(known)
        if unknown:
            sys.exit(f'Unknown test {name}(s) {sorted(unknown)}. Available: {known}')

    seqs = defaultdict(list)
    for k in meshes:
        seqs[(k[0], k[1])].append(k)
    for ks in seqs.values():
        ks.sort(key=frame_of)

    # Held-out frames of the training subjects: the official COMA test list, or
    # (no list) the last seen_test_fraction of every sequence.
    listed = sp.get('official_test_list', '').strip()
    if listed:
        with open(abspath(listed)) as fh:
            held = {tuple(line.strip().split('/')) for line in fh if line.strip()}
        missing = held - set(meshes)
        if missing:
            sys.exit(f'{len(missing)} meshes of {listed} not found under data_dir, '
                     f'e.g. {"/".join(sorted(missing)[0])}. Is the download complete?')
    else:
        frac = sp.getfloat('seen_test_fraction', 0.1)
        held = {k for ks in seqs.values() for k in ks[len(ks) - max(1, round(frac * len(ks))):]}

    # Training (subject, sequence) pairs, optionally a random subset in which
    # every sequence type keeps at least min_subjects_per_expression subjects.
    rng = random.Random(sp.getint('seed'))
    s_train = [s for s in subjects if s not in s_test]
    e_train = [e for e in expressions if e not in e_test]
    pairs = {(s, e) for s in s_train for e in e_train if (s, e) in seqs}
    frac = sp.getfloat('train_pair_fraction', 1.0)
    if frac < 1.0:
        k_min = sp.getint('min_subjects_per_expression', 5)
        keep = set()
        for e in e_train:
            have = [s for s in s_train if (s, e) in pairs]
            keep |= {(s, e) for s in rng.sample(have, min(k_min, len(have)))}
        rest = sorted(pairs - keep)
        rng.shuffle(rest)
        keep |= set(rest[:max(0, round(frac * len(pairs)) - len(keep))])
        pairs = keep

    train = sorted(k for k in meshes if (k[0], k[1]) in pairs and k not in held)

    def cell_of(k):
        return {(False, False): 'seen_subject', (True, False): 'unseen_subject',
                (False, True): 'unseen_expression', (True, True): 'unseen_both'}[
            (k[0] in s_test, k[1] in e_test)]

    stride = sp.getint('unseen_frame_stride', 5)
    cells = {c: [] for c in CELLS}
    cells['seen_subject'] = sorted(k for k in held
                                   if cell_of(k) == 'seen_subject' and (k[0], k[1]) in pairs)
    for (s, e), ks in sorted(seqs.items()):
        c = cell_of(ks[0])
        if c != 'seen_subject':
            cells[c] += ks[::stride]
    cells = {c: v for c, v in cells.items() if v}

    spec = dict(test_subjects=sorted(s_test), test_sequences=sorted(e_test),
                train_subjects=s_train, train_sequences=e_train,
                train_pairs=len(pairs), train_pair_fraction=frac,
                held_out_frames=listed or f'last {sp.getfloat("seen_test_fraction", 0.1):g} of each sequence',
                unseen_frame_stride=stride, seed=sp.getint('seed'),
                n_train_val=len(train), n_test={c: len(v) for c, v in cells.items()})
    return train, cells, spec, held, s_test, e_test


def fingerprint(train, cells):
    h = hashlib.sha1()
    for k in train:
        h.update('/'.join(k).encode())
    for c in sorted(cells):
        h.update(c.encode())
        for k in cells[c]:
            h.update('/'.join(k).encode())
    return h.hexdigest()


def load_verts(path):
    return np.asarray(load_mesh(path, process=False).vertices, dtype=np.float64)


def vert_hash(v):
    return hashlib.sha1(np.round(v, 5).astype(np.float32).tobytes()).hexdigest()


def check(name, ok, detail):
    print(f'  [{"OK" if ok else "FAIL"}] {name}: {detail}')
    return ok


def leakage_checks(meshes, train_paths, train, cells, held, s_test, e_test, val_seed, content=True):
    """Split checks, plus (content=True) a vertex-hash check and the mean of the
    exact training part that process() will use, for the processed-data check."""
    print('\nLeakage checks (split)')
    tests = sorted({k for v in cells.values() for k in v})
    ok = check('no test subject/sequence in training',
               not any(k[0] in s_test or k[1] in e_test for k in train),
               f'{len(train)} training meshes checked')
    ok &= check('no held-out test frame in training', not (set(train) & held),
                f'{len(set(train) & held)} found')
    ok &= check('no test mesh in training', not (set(train) & set(tests)),
                f'{len(set(train) & set(tests))} found')
    train_mean = None
    if content:
        # process() shuffles the sorted file list with random.seed(val_seed) and
        # keeps the first 80% for training; replay that to get its exact mean.
        idx = list(range(len(train_paths)))
        random.Random(val_seed).shuffle(idx)
        train_part = set(idx[:int(0.8 * len(idx))])
        print(f'  loading {len(train)} training + {len(tests)} test meshes for the content check ...',
              flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            total, hashes = 0.0, set()
            for i, v in enumerate(ex.map(load_verts, train_paths)):
                if i in train_part:
                    total = total + v
                hashes.add(vert_hash(v))
            train_mean = total / len(train_part)
            dup = sum(vert_hash(v) in hashes for v in ex.map(load_verts, [meshes[k] for k in tests]))
        ok &= check('no test mesh identical to a training mesh', dup == 0,
                    f'{dup} identical vertex sets')
    return ok, train_mean


def processed_checks(config, n_train, train_mean):
    print('\nLeakage checks (processed data)')
    proc = os.path.join(config['dataset_dir'], 'processed_completion')
    n = {s: len(torch.load(os.path.join(proc, f'{s}.pt'), weights_only=False)[1]['x']) - 1
         for s in ('training', 'eval')}
    ok = check('cache built from the split only', n['training'] + n['eval'] == n_train,
               f'train {n["training"]} + val {n["eval"]} vs {n_train} in train.txt')
    mean = torch.load(os.path.join(proc, 'norm.pt'), weights_only=False)['mean'].double().numpy()
    if train_mean is not None:
        d = np.abs(mean - train_mean).max() * 1000
        ok &= check('normalisation mean = mean of the training part', d < 1e-3,
                    f'max diff {d:.6f} mm')
    d = np.abs(load_verts(config['template_file']) - mean).max() * 1000
    ok &= check('template = training mean', d < 1e-3, f'max diff {d:.6f} mm')
    return ok


def build_benchmark(files, out_dir, bm, seed):
    os.makedirs(out_dir, exist_ok=True)
    strategies = csv_list(bm.get('strategies', 'random_scatter, bfs_patch, plane_cut'))
    lo, hi = bm.getfloat('ratio_min', 0.05), bm.getfloat('ratio_max', 0.40)
    first = load_mesh(files[0][1], process=False)
    faces = np.asarray(first.faces)
    adj = _build_adjacency(faces, len(first.vertices))
    rng = np.random.RandomState(seed)
    masks, rows = [], []
    for key, path in files:
        verts = np.asarray(load_mesh(path, process=False).vertices, dtype=np.float32)
        for strat in strategies:
            ratio = rng.uniform(lo, hi)
            m = np.asarray(MASK_STRATEGIES[strat](verts, faces, mask_ratio=ratio, rng=rng, adj=adj),
                           dtype=bool)
            masks.append(m)
            rows.append(dict(row=len(rows), subject=key[0], expression=key[1],
                             frame=f'{frame_of(key):06d}', gt_path=os.path.abspath(path),
                             strategy=strat, target_ratio=round(ratio, 4),
                             actual_ratio=round(float(m.mean()), 4)))
    np.savez_compressed(os.path.join(out_dir, 'masks.npz'), masks=np.stack(masks))
    with open(os.path.join(out_dir, 'index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'coma.cfg'))
    ap.add_argument('--skip_content_check', action='store_true',
                    help='skip the vertex-hash and exact-mean checks (saves a few minutes)')
    args = ap.parse_args()

    cp = configparser.RawConfigParser()
    cp.read(args.config)
    config = read_config(args.config)
    work = config['work_dir']
    io = cp['I/O parameters']
    data_dirs = [abspath(d) for d in csv_list(io.get('data_dir', io.get('coma_dir', '')))]
    split_dir = os.path.join(work, 'splits')

    print('config  :', args.config)
    print('data_dir:', ', '.join(data_dirs))
    print('work_dir:', work)
    for i, d in enumerate(data_dirs):
        h5 = sorted(glob(os.path.join(d, 'registrations_*.hdf5')))
        if h5:
            data_dirs[i] = os.path.join(work, 'meshes')
            export_hdf5(h5, data_dirs[i], io.getint('hdf5_frame_stride', 6))
    meshes = scan(data_dirs)
    if not meshes:
        sys.exit('No meshes found. Expected <data_dir>/<subject>/<sequence>/*.ply, '
                 '<data_dir>/<subject>_<sequence>/*.ply or registrations_*.hdf5')
    print(f'found {len(meshes)} meshes, {len({k[0] for k in meshes})} subjects, '
          f'{len({k[1] for k in meshes})} sequence types')

    # ---- 1. split ----
    train, cells, spec, held, s_test, e_test = make_split(meshes, cp['split'])
    spec['fingerprint'] = fingerprint(train, cells)
    spec_path = os.path.join(split_dir, 'split.json')
    if os.path.exists(spec_path):
        with open(spec_path) as fh:
            old = json.load(fh)['fingerprint']
        if old != spec['fingerprint']:
            sys.exit(f'The split in the config differs from the one already prepared in {work}.\n'
                     f'Delete {work} (or set another work_dir) and run again.')
    os.makedirs(split_dir, exist_ok=True)
    with open(spec_path, 'w') as fh:
        json.dump(spec, fh, indent=2)
    train_paths = sorted(os.path.abspath(meshes[k]) for k in train)
    with open(config['train_list'], 'w') as fh:
        fh.write('\n'.join(train_paths) + '\n')
    for c, ks in cells.items():
        with open(os.path.join(split_dir, f'test_{c}.txt'), 'w') as fh:
            fh.write('\n'.join('/'.join(k) for k in ks) + '\n')

    print('\nSplit')
    print('  test subjects :', ', '.join(spec['test_subjects']))
    print('  test sequences:', ', '.join(spec['test_sequences']) or '(none: subject split)')
    print(f'  train + val   : {len(train)} meshes '
          f'({len(spec["train_subjects"])} subjects, {spec["train_pairs"]} subject-sequence pairs)')
    for c, ks in cells.items():
        print(f'  test {c:17s}: {len(ks)} meshes')

    val_seed = cp.getint('split', 'val_seed', fallback=1)
    ok, train_mean = leakage_checks(meshes, train_paths, train, cells, held, s_test, e_test,
                                    val_seed, content=not args.skip_content_check)
    if not ok:
        sys.exit('\nLEAKAGE CHECK FAILED, nothing was processed.')

    # ---- 2. processed cache (80/20 train/val shuffle seeded with val_seed) ----
    print('\nProcessing (skipped if the cache already exists) ...', flush=True)
    random.seed(val_seed)
    np.random.seed(val_seed)
    torch.manual_seed(val_seed)
    from utils.completion_dataset import MeshCompletionDataset
    MeshCompletionDataset(config, 'train')
    if not processed_checks(config, len(train), train_mean):
        sys.exit('\nPROCESSED-DATA CHECK FAILED. Delete work_dir and run again.')

    # ---- 3. benchmark ----
    print('\nBenchmark')
    for i, c in enumerate(CELLS):
        if c not in cells:
            continue
        out = os.path.join(work, 'benchmark', c)
        if os.path.exists(os.path.join(out, 'index.csv')):
            print(f'  {c}: already built')
            continue
        n = build_benchmark([(k, meshes[k]) for k in cells[c]], out, cp['benchmark'],
                            cp.getint('split', 'seed') + i)
        print(f'  {c}: {n} samples')

    default = os.path.join(ROOT, 'config', 'coma.cfg')
    print('\nDone. Next: python train.py'
          + ('' if os.path.abspath(args.config) == default else f' --config {args.config}'))


if __name__ == '__main__':
    main()
