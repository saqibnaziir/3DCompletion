"""
Exp2: check the processed cache after process_official_split.py has built it.

The normalisation mean/std and template.ply are derived from the training data.
If they were computed with the held-out subjects/expressions included, the
"unseen" cells would not be unseen. This verifies that:
    1. every path in the config resolves inside revision/exp2_subj_expr/
    2. train + val in processed_completion/ == number of files in raw_train/
    3. the stored mean equals the mean of the raw_train meshes only
    4. template.ply equals that mean (and differs from revision/data/template.ply)

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/exp2_subj_expr/check_processed.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # same as train_diffusion.py
import sys
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from trimesh import load_mesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from config.config import read_config

CFG = os.path.join(_ROOT, 'config', 'config_diffusion_exp2_subjexpr.cfg')
EXP = os.path.join(_ROOT, 'revision', 'exp2_subj_expr')


def verts(path):
    return np.asarray(load_mesh(path, process=False).vertices, dtype=np.float64)


def main():
    config = read_config(CFG)
    ok = True
    for k in ('dataset_dir', 'template_file', 'checkpoint_dir', 'train_dir'):
        inside = os.path.abspath(config[k]).startswith(EXP + os.sep)
        print(f'  [1] {k:15s} {config[k]}', 'OK' if inside else 'FAIL')
        ok &= inside

    proc = os.path.join(config['dataset_dir'], 'processed_completion')
    n = {s: len(torch.load(os.path.join(proc, f'{s}.pt'), weights_only=False)[1]['x']) - 1
         for s in ('training', 'eval')}
    files = sorted(glob(os.path.join(config['train_dir'], '*', '*', '*.ply')))
    same = n['training'] + n['eval'] == len(files)
    print(f'  [2] processed train {n["training"]} + val {n["eval"]} = '
          f'{n["training"] + n["eval"]}, raw_train {len(files)}', 'OK' if same else 'FAIL')
    ok &= same

    # process() takes the mean over its 80% training part only, so it differs
    # slightly from the mean of all of raw_train. A mean that included the
    # held-out subjects/expressions would be off by millimetres, not 0.05 mm.
    norm = torch.load(os.path.join(proc, 'norm.pt'), weights_only=False)
    mean = norm['mean'].double().numpy()
    with ThreadPoolExecutor(max_workers=8) as ex:   # processes crash: torch + OpenMP in workers
        raw_mean = np.mean(list(ex.map(verts, files)), axis=0)
    d_raw = np.abs(mean - raw_mean).max() * 1000
    good = d_raw < 0.5            # mm; 80% subset of the same cells vs all of them
    print(f'  [3] stored mean vs raw_train mean: max diff {d_raw:.4f} mm', 'OK' if good else 'FAIL')
    ok &= good

    tpl = verts(config['template_file'])
    d_tpl = np.abs(tpl - mean).max() * 1000
    print(f'  [4] template.ply vs stored mean: max diff {d_tpl:.4f} mm',
          'OK' if d_tpl < 1e-3 else 'FAIL')
    ok &= d_tpl < 1e-3
    old_tpl = os.path.join(_ROOT, 'revision', 'data', 'template.ply')
    if os.path.exists(old_tpl):
        d_old = np.abs(verts(old_tpl) - tpl).max() * 1000
        print(f'      vs revision/data/template.ply (official split): max diff {d_old:.4f} mm '
              '(should be > 0: a different training set gives a different mean)')

    print('\nall processed-data checks passed' if ok else '\nPROCESSED-DATA CHECK FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
