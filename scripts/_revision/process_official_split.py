"""
Build the processed cache for the revision (official COMA train/test split).

Reads raw meshes from `train_dir` (data/train, official test subset removed) and
writes everything into the revision root (revision/data): processed_completion/,
norm_completion.pt, template.ply and completion_test_samples/. The old data/
artefacts are not touched.

The 80/20 shuffle inside MeshCompletionDataset.process() is now train/VALIDATION
(checkpoint selection only); the test set is data/COMA_test_subset/gt.

Usage (from D:\\COURSAI\\3Dis, env 3dmesh):
    python scripts/_revision/process_official_split.py
"""
import os
import sys
import random
import argparse

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from config.config import read_config
from utils.completion_dataset import MeshCompletionDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file',
                    default=os.path.join(_ROOT, 'config', 'config_diffusion_revision.cfg'))
    ap.add_argument('--seed', type=int, default=1,
                    help='same default as train_diffusion.py, so the split matches')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = read_config(args.config_file)
    assert os.path.abspath(config['dataset_dir']) != os.path.join(_ROOT, 'data'), \
        'revision config must not point at the old data/ root'
    print('dataset_dir :', config['dataset_dir'])
    print('train_dir   :', config.get('train_dir'))
    print('template    :', config['template_file'])

    ds_train = MeshCompletionDataset(config, 'train')
    ds_eval = MeshCompletionDataset(config, 'eval')
    print(f'train={len(ds_train)}  val={len(ds_eval)}  '
          f'verts/mesh={ds_train.get(0).x.shape[0]}')


if __name__ == '__main__':
    main()
