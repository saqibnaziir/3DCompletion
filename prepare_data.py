"""
Step 1 of 3: prepare the data.

    python prepare_data.py [--config config/coma.cfg]

Reads the raw COMA meshes from `coma_dir`, and writes into `work_dir`:
    splits/split.json, train.txt, test_<cell>.txt   the split
    data/                                           processed cache, template, normalisation
    benchmark/<cell>/masks.npz, index.csv           fixed test holes, one mask per strategy

Test cells (empty ones are skipped):
    seen_subject       training subjects, official COMA test frames (reference)
    unseen_subject     test subjects, training expressions
    unseen_expression  training subjects, test expressions
    unseen_both        test subjects, test expressions

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


def scan(dirs):
    """{(subject, expression, filename): path} for <dir>/<subject>/<expression>/*.ply."""
    meshes = {}
    for d in dirs:
        for f in sorted(glob(os.path.join(d, '*', '*', '*.ply'))):
            parts = os.path.normpath(f).split(os.sep)
            meshes.setdefault((parts[-3], parts[-2], parts[-1]), f)
    return meshes


def frame_of(key):
    return int(key[2].split('.')[1])


def make_split(meshes, sp):
    subjects = sorted({k[0] for k in meshes})
    expressions = sorted({k[1] for k in meshes})
    s_test = set(csv_list(sp.get('test_subjects', '')))
    e_test = set(csv_list(sp.get('test_expressions', '')))
    for name, given, known in (('subject', s_test, subjects), ('expression', e_test, expressions)):
        unknown = given - set(known)
        if unknown:
            sys.exit(f'Unknown test {name}(s) {sorted(unknown)}. Available: {known}')

    with open(abspath(sp['official_test_list'])) as fh:
        official = {tuple(line.strip().split('/')) for line in fh if line.strip()}
    missing = official - set(meshes)
    if missing:
        sys.exit(f'{len(missing)} official test meshes not found under coma_dir, '
                 f'e.g. {"/".join(sorted(missing)[0])}. Is the COMA download complete?')

    # Training (subject, expression) pairs, optionally a random subset in which
    # every expression keeps at least min_subjects_per_expression subjects.
    rng = random.Random(sp.getint('seed'))
    s_train = [s for s in subjects if s not in s_test]
    e_train = [e for e in expressions if e not in e_test]
    pairs = {(s, e) for s in s_train for e in e_train}
    frac = sp.getfloat('train_pair_fraction', 1.0)
    if frac < 1.0:
        k_min = sp.getint('min_subjects_per_expression', 5)
        keep = set()
        for e in e_train:
            keep |= {(s, e) for s in rng.sample(s_train, min(k_min, len(s_train)))}
        rest = sorted(pairs - keep)
        rng.shuffle(rest)
        keep |= set(rest[:max(0, round(frac * len(pairs)) - len(keep))])
        pairs = keep

    train = sorted(k for k in meshes if (k[0], k[1]) in pairs and k not in official)

    def cell_of(k):
        return {(False, False): 'seen_subject', (True, False): 'unseen_subject',
                (False, True): 'unseen_expression', (True, True): 'unseen_both'}[
            (k[0] in s_test, k[1] in e_test)]

    stride = sp.getint('unseen_frame_stride', 5)
    seqs = defaultdict(list)
    for k in meshes:
        seqs[(k[0], k[1])].append(k)
    cells = {c: [] for c in CELLS}
    cells['seen_subject'] = sorted(k for k in official
                                   if cell_of(k) == 'seen_subject' and (k[0], k[1]) in pairs)
    for (s, e), ks in sorted(seqs.items()):
        c = cell_of(ks[0])
        if c != 'seen_subject':
            cells[c] += sorted(ks, key=frame_of)[::stride]
    cells = {c: v for c, v in cells.items() if v}

    spec = dict(test_subjects=sorted(s_test), test_expressions=sorted(e_test),
                train_subjects=s_train, train_expressions=e_train,
                train_pairs=len(pairs), train_pair_fraction=frac,
                unseen_frame_stride=stride, seed=sp.getint('seed'),
                n_train_val=len(train), n_test={c: len(v) for c, v in cells.items()})
    return train, cells, spec, official, s_test, e_test


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


def leakage_checks(meshes, train, cells, official, s_test, e_test, content=True):
    print('\nLeakage checks (split)')
    tests = sorted({k for v in cells.values() for k in v})
    ok = check('no test subject/expression in training',
               not any(k[0] in s_test or k[1] in e_test for k in train),
               f'{len(train)} training meshes checked')
    ok &= check('no official test frame in training', not (set(train) & official),
                f'{len(set(train) & official)} found')
    ok &= check('no test mesh in training', not (set(train) & set(tests)),
                f'{len(set(train) & set(tests))} found')
    train_mean = None
    if content:
        print(f'  loading {len(train)} training + {len(tests)} test meshes for the content check ...',
              flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            total, hashes = None, set()
            for v in ex.map(load_verts, [meshes[k] for k in train]):
                total = v if total is None else total + v
                hashes.add(vert_hash(v))
            train_mean = total / len(train)
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
        # process() averages its 80% training part, so a small difference is
        # expected; a mean that included test meshes would be millimetres off.
        d = np.abs(mean - train_mean).max() * 1000
        ok &= check('normalisation mean from training meshes', d < 0.5, f'max diff {d:.3f} mm')
    d = np.abs(load_verts(config['template_file']) - mean).max() * 1000
    ok &= check('template = training mean', d < 1e-3, f'max diff {d:.5f} mm')
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
    for (subj, expr, fname), path in files:
        verts = np.asarray(load_mesh(path, process=False).vertices, dtype=np.float32)
        for strat in strategies:
            ratio = rng.uniform(lo, hi)
            m = np.asarray(MASK_STRATEGIES[strat](verts, faces, mask_ratio=ratio, rng=rng, adj=adj),
                           dtype=bool)
            masks.append(m)
            rows.append(dict(row=len(rows), subject=subj, expression=expr,
                             frame=fname.split('.')[1], gt_path=os.path.abspath(path),
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
                    help='skip the vertex-hash leakage check (saves a few minutes)')
    args = ap.parse_args()

    cp = configparser.RawConfigParser()
    cp.read(args.config)
    config = read_config(args.config)
    work = config['work_dir']
    coma_dirs = [abspath(d) for d in csv_list(cp.get('I/O parameters', 'coma_dir'))]
    split_dir = os.path.join(work, 'splits')

    print('config  :', args.config)
    print('coma_dir:', ', '.join(coma_dirs))
    print('work_dir:', work)
    meshes = scan(coma_dirs)
    if not meshes:
        sys.exit('No meshes found. Expected <coma_dir>/<subject>/<expression>/*.ply')
    print(f'found {len(meshes)} meshes, {len({k[0] for k in meshes})} subjects, '
          f'{len({k[1] for k in meshes})} expressions')

    # ---- 1. split ----
    train, cells, spec, official, s_test, e_test = make_split(meshes, cp['split'])
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
    with open(config['train_list'], 'w') as fh:
        fh.write('\n'.join(os.path.abspath(meshes[k]) for k in train) + '\n')
    for c, ks in cells.items():
        with open(os.path.join(split_dir, f'test_{c}.txt'), 'w') as fh:
            fh.write('\n'.join('/'.join(k) for k in ks) + '\n')

    print('\nSplit')
    print('  test subjects   :', ', '.join(spec['test_subjects']))
    print('  test expressions:', ', '.join(spec['test_expressions']) or '(none: subject split)')
    print(f'  train + val     : {len(train)} meshes '
          f'({len(spec["train_subjects"])} subjects, {spec["train_pairs"]} subject-expression pairs)')
    for c, ks in cells.items():
        print(f'  test {c:17s}: {len(ks)} meshes')

    ok, train_mean = leakage_checks(meshes, train, cells, official, s_test, e_test,
                                    content=not args.skip_content_check)
    if not ok:
        sys.exit('\nLEAKAGE CHECK FAILED, nothing was processed.')

    # ---- 2. processed cache (train/val shuffle seeded like train_diffusion.py) ----
    print('\nProcessing (skipped if the cache already exists) ...', flush=True)
    seed = cp.getint('split', 'val_seed', fallback=1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    print(f'\nDone. Next: python train.py' + ('' if args.config.endswith(os.path.join('config', 'coma.cfg'))
                                            else f' --config {args.config}'))


if __name__ == '__main__':
    main()
