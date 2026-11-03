"""
Launcher: train FaceCom's generator on our COMA split, without editing their repo.

It only does the plumbing their scripts/train.py cannot do for itself:
  * puts this directory first on sys.path, so `quad_mesh_simplify.py` here
    satisfies their import (see that file for why it must be a real module),
  * puts FaceCom_repo ahead of 3Dis on sys.path, since 3Dis also ships utils/
    and config/ packages and the wrong one silently loads a different network,
  * seeds template_file if absent, because get_mesh_matrices() reads it on runs
    where MyDataset.process() is skipped (i.e. every resume),
  * prints the effective settings so the run is self-documenting in the log.

Training itself is their train() verbatim: Adam, L1 + MSE + spherical latent
regularization, StepLR(30, 0.5), best-on-eval checkpointing to checkpoint_dir.

Usage (conda env: 3dmesh)
  python stage_coma_trainset.py            # once, writes coma_dataset/train
  python train_facecom_coma.py --config_file config_facecom_coma.cfg
"""

import os
import sys
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
FACECOM_ROOT = os.path.abspath(os.path.join(HERE, "..", "FaceCom_repo"))
DIS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, HERE)           # our quad_mesh_simplify.py
sys.path.insert(1, FACECOM_ROOT)   # must precede 3Dis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file",
                    default=os.path.join(HERE, "config_facecom_coma.cfg"))
    args = ap.parse_args()

    from config.config import read_config
    config = read_config(args.config_file)

    train_dir = os.path.join(config["dataset_dir"], "train")
    if not os.path.isdir(train_dir):
        sys.exit(f"ABORT: {train_dir} does not exist.\n"
                 f"  Run:  python stage_coma_trainset.py")
    n_meshes = len([f for f in os.listdir(train_dir) if f.endswith(".ply")])
    if n_meshes == 0:
        sys.exit(f"ABORT: no .ply files in {train_dir}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    if not os.path.exists(config["template_file"]):
        os.makedirs(os.path.dirname(config["template_file"]), exist_ok=True)
        shutil.copy(os.path.join(DIS_ROOT, "data", "template.ply"),
                    config["template_file"])
        print(f"seeded template_file <- data/template.ply "
              f"(MyDataset.process() will overwrite it with the training mean)")

    import torch
    import _shim
    _shim.allow_full_unpickle()   # PyTorch 2.6 weights_only default vs PyG collates

    print("=" * 66)
    print("FaceCom generator on COMA")
    print("=" * 66)
    print(f"  repo         : {FACECOM_ROOT}")
    print(f"  dataset_dir  : {config['dataset_dir']}  ({n_meshes} meshes)")
    print(f"  checkpoints  : {config['checkpoint_dir']}")
    print(f"  epochs       : {config['epoch']}   batch {config['batch_size']}   "
          f"lr {config['lr']}   workers {config['num_workers']}")
    print(f"  device       : {'cuda' if torch.cuda.is_available() else 'CPU (!)'}")
    print("=" * 66)

    sys.path.insert(0, os.path.join(FACECOM_ROOT, "scripts"))
    from scripts.train import train
    train(config)


if __name__ == "__main__":
    main()
