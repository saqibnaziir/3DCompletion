"""
Exp2: (subject, expression) split for COMA -- build the split, the training view,
the per-condition test benchmarks, and check for leakage.

Hold out a set of test subjects S_test AND a set of test expressions E_test at
the same time, train on S_train x E_train only, and one checkpoint gives all
three conditions:

    ref : S_train x E_train  official test frames (x91-x00), seen/seen reference
    a   : S_test  x E_train  unseen subject,  seen expression
    b   : S_train x E_test   seen subject,    unseen expression
    c   : S_test  x E_test   both unseen

Everything is written under revision/exp2_subj_expr/; data/ and revision/data
are never written to.

    raw_train/<subject>/<expr>/*.ply   hard links into data/train (no copy)
    splits/split.json, splits/*.txt    the split, one file list per set
    benchmark/<cell>/                  masks + index, same format as
                                       revision/benchmark, so
                                       eval_official_test.py runs with --bench_dir

Leakage checks (all must pass, otherwise the script exits non-zero):
    1. metadata  : no training file has a test subject or a test expression
    2. file      : no test file is the same file (inode) as a training file
    3. content   : no test mesh has the same vertices (rounded 1e-5) as any
                   training mesh

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/exp2_subj_expr/make_split.py
    python scripts/_revision/exp2_subj_expr/make_split.py --check_only
"""
import os
import sys
import csv
import json
import hashlib
import argparse
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from trimesh import load_mesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from utils.masking import MASK_STRATEGIES, _build_adjacency

STRATEGIES = ('random_scatter', 'bfs_patch', 'plane_cut')
CELLS = ('ref', 'a', 'b', 'c')


def parse(path):
    """(subject, expression, frame) from .../<subject>/[<expr>/]<expr>.<frame>.ply."""
    parts = os.path.normpath(path).split(os.sep)
    expr, frame = parts[-1].split('.')[:2]
    subj = parts[-2] if parts[-2] != expr else parts[-3]
    if not subj.startswith('FaceTalk_'):          # COMA_test_subset drops the prefix
        subj = 'FaceTalk_' + subj
    return subj, expr, int(frame)


def vert_hash(path):
    v = np.asarray(load_mesh(path, process=False).vertices, dtype=np.float64)
    return hashlib.sha1(np.round(v, 5).astype(np.float32).tobytes()).hexdigest()


def hash_all(paths, desc):
    print(f'  hashing {len(paths)} {desc} meshes ...', flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        return dict(zip(paths, ex.map(vert_hash, paths, chunksize=64)))


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_bench(files, out_dir, seed):
    """Same masks/format as scripts/_revision/build_test_benchmark.py."""
    os.makedirs(out_dir, exist_ok=True)
    first = load_mesh(files[0], process=False)
    faces = np.asarray(first.faces)
    n_verts = len(first.vertices)
    adj = _build_adjacency(faces, n_verts)
    rng = np.random.RandomState(seed)
    masks, rows = [], []
    for f in files:
        verts = np.asarray(load_mesh(f, process=False).vertices, dtype=np.float32)
        assert len(verts) == n_verts, f
        subj, expr, frame = parse(f)
        for strat in STRATEGIES:
            ratio = rng.uniform(0.05, 0.40)
            m = MASK_STRATEGIES[strat](verts, faces, mask_ratio=ratio, rng=rng, adj=adj)
            masks.append(np.asarray(m, dtype=bool))
            rows.append(dict(row=len(rows), subject=subj, expression=expr,
                             frame=f'{frame:06d}', gt_path=os.path.relpath(f, _ROOT),
                             strategy=strat, target_ratio=round(ratio, 4),
                             actual_ratio=round(float(m.mean()), 4)))
    # File names kept identical to revision/benchmark so eval_official_test.py
    # can be pointed at this folder with --bench_dir.
    np.savez_compressed(os.path.join(out_dir, 'official_test_masks.npz'), masks=np.stack(masks))
    with open(os.path.join(out_dir, 'official_test_index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_src', default=os.path.join(_ROOT, 'data', 'train'))
    ap.add_argument('--test_src', default=os.path.join(_ROOT, 'data', 'COMA_test_subset', 'gt'))
    ap.add_argument('--exp_root', default=os.path.join(_ROOT, 'revision', 'exp2_subj_expr'))
    ap.add_argument('--n_test_subjects', type=int, default=2)
    ap.add_argument('--test_subjects', default=None,
                    help='comma list; default = random draw with --seed')
    ap.add_argument('--test_expressions', default='high_smile,mouth_extreme',
                    help='one moderate + one hard expression')
    ap.add_argument('--stride', default='a=5,b=5,c=2',
                    help='keep every k-th frame per sequence in the unseen cells')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--check_only', action='store_true',
                    help='re-run the leakage checks on an existing split')
    ap.add_argument('--skip_content_check', action='store_true')
    args = ap.parse_args()

    root = os.path.abspath(args.exp_root)
    for forbidden in (os.path.join(_ROOT, 'data'), os.path.join(_ROOT, 'revision', 'data')):
        assert root != forbidden and not root.startswith(forbidden + os.sep), \
            f'exp_root must not be inside {forbidden}'
    raw_train = os.path.join(root, 'raw_train')
    split_dir = os.path.join(root, 'splits')
    split_json = os.path.join(split_dir, 'split.json')

    # ---------------- the split ----------------
    train_files = sorted(glob(os.path.join(args.train_src, '*', '*', '*.ply')))
    test_files = sorted(glob(os.path.join(args.test_src, '*', '*.ply')))
    subjects = sorted({parse(f)[0] for f in train_files})
    expressions = sorted({parse(f)[1] for f in train_files})
    print(f'pool: {len(train_files)} train + {len(test_files)} official-test meshes, '
          f'{len(subjects)} subjects x {len(expressions)} expressions')

    if args.check_only:
        with open(split_json) as fh:
            spec = json.load(fh)
        s_test, e_test = set(spec['test_subjects']), set(spec['test_expressions'])
    else:
        if args.test_subjects:
            s_test = set(args.test_subjects.split(','))
        else:
            rng = np.random.RandomState(args.seed)
            s_test = set(rng.choice(subjects, args.n_test_subjects, replace=False).tolist())
        e_test = set(args.test_expressions.split(','))
    assert s_test <= set(subjects), f'unknown subject in {s_test}'
    assert e_test <= set(expressions), f'unknown expression in {e_test}'
    print('test subjects   :', sorted(s_test))
    print('test expressions:', sorted(e_test))

    def cell_of(f):
        s, e, _ = parse(f)
        return {(False, False): 'ref', (True, False): 'a',
                (False, True): 'b', (True, True): 'c'}[(s in s_test, e in e_test)]

    strides = {k: int(v) for k, v in (kv.split('=') for kv in args.stride.split(','))}
    train_keep = [f for f in train_files if cell_of(f) == 'ref']

    # Unseen cells may use every frame of their sequences (train pool + official
    # test frames), subsampled per sequence. The seen/seen reference uses only
    # the official test frames, which never appear in data/train.
    seqs = defaultdict(dict)
    for f in train_files + test_files:
        s, e, fr = parse(f)
        seqs[(s, e)].setdefault(fr, f)
    cells = {'ref': [f for f in test_files if cell_of(f) == 'ref']}
    for c in ('a', 'b', 'c'):
        cells[c] = []
        for (s, e), frames in sorted(seqs.items()):
            if cell_of(frames[min(frames)]) != c:
                continue
            cells[c] += [frames[k] for k in sorted(frames)[::strides[c]]]

    rel = lambda f: os.path.relpath(f, _ROOT).replace(os.sep, '/')
    link_path = lambda f: os.path.join(raw_train, *os.path.normpath(f).split(os.sep)[-3:])

    if not args.check_only:
        os.makedirs(split_dir, exist_ok=True)
        # training view: hard links, nothing copied, data/train untouched
        existing = set(glob(os.path.join(raw_train, '*', '*', '*.ply')))
        wanted = {link_path(f) for f in train_keep}
        extra = existing - wanted
        assert not extra, (f'{len(extra)} unexpected files in {raw_train} '
                           f'(e.g. {next(iter(extra))}); delete raw_train and rerun')
        todo = [f for f in train_keep if link_path(f) not in existing]
        print(f'raw_train: linking {len(todo)} files ({len(existing)} already present)')
        for f in todo:
            link_or_copy(f, link_path(f))

        with open(os.path.join(split_dir, 'train.txt'), 'w') as fh:
            fh.write('\n'.join(rel(f) for f in train_keep) + '\n')
        for c in CELLS:
            with open(os.path.join(split_dir, f'test_{c}.txt'), 'w') as fh:
                fh.write('\n'.join(rel(f) for f in cells[c]) + '\n')
        with open(split_json, 'w') as fh:
            json.dump(dict(seed=args.seed, test_subjects=sorted(s_test),
                           test_expressions=sorted(e_test),
                           train_subjects=sorted(set(subjects) - s_test),
                           train_expressions=sorted(set(expressions) - e_test),
                           strides=strides, n_train_val=len(train_keep),
                           n_test={c: len(cells[c]) for c in CELLS}), fh, indent=2)

    # ---------------- leakage checks ----------------
    print('\nleakage checks')
    ok = True
    linked = sorted(glob(os.path.join(raw_train, '*', '*', '*.ply')))

    # 1. metadata
    bad = [f for f in linked if parse(f)[0] in s_test or parse(f)[1] in e_test]
    print(f'  [1] metadata : {len(bad)} training files with a test subject/expression',
          'OK' if not bad else 'FAIL')
    ok &= not bad
    for c in CELLS:
        wrong = [f for f in cells[c] if cell_of(f) != c]
        ok &= not wrong
        if wrong:
            print(f'      FAIL: {len(wrong)} files in cell {c} belong elsewhere')
    if len(linked) != len(train_keep):
        print(f'      FAIL: raw_train has {len(linked)} files, expected {len(train_keep)}')
        ok = False

    # 2. same file on disk
    ino = lambda f: (os.stat(f).st_dev, os.stat(f).st_ino)
    train_inodes = {ino(f) for f in linked}
    shared = [f for c in CELLS for f in cells[c] if ino(f) in train_inodes]
    print(f'  [2] file     : {len(shared)} test files are the same file as a training file',
          'OK' if not shared else 'FAIL')
    ok &= not shared

    # 3. identical vertex content
    if not args.skip_content_check:
        test_all = sorted({f for c in CELLS for f in cells[c]})
        h_train = hash_all(linked, 'training')
        h_test = hash_all(test_all, 'test')
        train_set = set(h_train.values())
        dup = [f for f in test_all if h_test[f] in train_set]
        print(f'  [3] content  : {len(dup)} test meshes identical to a training mesh',
              'OK' if not dup else 'FAIL')
        for f in dup[:5]:
            print('      ', rel(f))
        ok &= not dup

    print('\ncounts')
    print(f'  train+val (S_train x E_train, 80/20 inside process()): {len(train_keep)}')
    for c in CELLS:
        print(f'  test_{c:3s}: {len(cells[c]):5d} meshes  -> {3 * len(cells[c])} benchmark samples')

    if not ok:
        print('\nLEAKAGE CHECK FAILED')
        sys.exit(1)
    print('\nall leakage checks passed')

    # ---------------- benchmarks ----------------
    if not args.check_only:
        for i, c in enumerate(CELLS):
            n = build_bench(cells[c], os.path.join(root, 'benchmark', c), args.seed + i)
            print(f'  benchmark/{c}: {n} samples')


if __name__ == '__main__':
    main()
