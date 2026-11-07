# 3DCompletion: Conditional Offset Diffusion for Registered Mesh Completion

Completes registered 3D face meshes (COMA, fixed topology) that have holes. A graph U-Net diffusion model predicts the missing vertices, and the observed vertices are kept exactly as given.

## 1. Install

```bash
conda env create -f environment.yml
conda activate 3dmesh
```
Tested with Python 3.10, PyTorch 2.6 (CUDA 12.4) and PyTorch Geometric 2.6.1.

## 2. Put the data in place

Download COMA (https://coma.is.tue.mpg.de/, licence required) and put the meshes in `data/COMA/`, keeping the original layout:

```
data/COMA/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply
data/COMA/FaceTalk_170725_00137_TA/cheeks_in/...
...                                  (12 subjects x 12 expressions, 20,461 meshes)
```

If your data is somewhere else, change `coma_dir` in `config/coma.cfg`.

## 3. Run

```bash
python prepare_data.py     # split, leakage checks, processed cache, test benchmark  (~15 min, CPU)
python train.py            # train the model                                          (~3 days on one GPU)
python test.py             # evaluate, print the results table
```

That's it. Each step reads `config/coma.cfg`. To resume an interrupted training run, use `python train.py --resume`.

## The config

All settings are in **[`config/coma.cfg`](config/coma.cfg)**, with a comment on each one. The ones you may want to change:

| Setting | Default | Meaning |
|---|---|---|
| `coma_dir` | `data/COMA` | where the raw meshes are |
| `work_dir` | `runs/coma_subject_split` | where everything generated goes |
| `test_subjects` | 2 subjects | held out from training |
| `test_expressions` | empty | also hold out expressions (empty = subject split) |
| `train_pair_fraction` | 1.0 | train on a random subset of (subject, expression) pairs |
| `run_name` | `full` | name of the checkpoint folder |
| `epoch`, `batch_size` | 300, 4 | training length and batch size |

For a different experiment, copy the file, change it (and give it its own `work_dir`), and pass it to all three commands: `python train.py --config config/my_exp.cfg`.

## The split and the test

- **Training:** every mesh of the training subjects, minus the official COMA test frames ([`splits/coma_official_test.txt`](splits/coma_official_test.txt)). 80% is used for training and 20% for validation, which only selects the checkpoint.
- **Test**, one table row per cell:
  - `seen_subject`: training subjects, official test frames (the reference)
  - `unseen_subject`: the held-out subjects (every 5th frame)
  - `unseen_expression` / `unseen_both`: only if `test_expressions` is set
- **Holes:** each test mesh gets three fixed holes (scattered vertices, a patch, a plane cut), 5–40% of the vertices, seed 2026. Every method must be scored on the same holes.
- **Leakage checks:** `prepare_data.py` stops if any test subject, expression or mesh (compared by vertex content) is in the training data, or if the normalisation and template are not computed from the training set.

## Outputs (in `work_dir`)

```
splits/                  split.json (the split definition), train.txt, test_<cell>.txt
data/                    processed cache, template.ply, normalisation
benchmark/<cell>/        masks.npz + index.csv (the fixed test holes)
checkpoints/predict_x0_<run_name>/   weights (best-val EMA), train_log.csv, TensorBoard logs
results/<run_name>/<cell>/per_sample.csv   error per test sample
results/<run_name>/summary.csv          the results table
```

## Repository layout

```
prepare_data.py, train.py, test.py   the three commands
config/coma.cfg                      the config  (config.py = parser)
splits/                              official COMA test list
scripts/                             training loop, sampler, evaluation (used by the commands)
utils/                               model, diffusion, dataset, masking, losses, metrics
Compattators/                        baselines (FaceCom, 3DGeoMeshNet)
research/                            paper experiments: ablations, figures, analysis,
                                     earlier configs (research/configs/), revision runs
CHANGELOG.md                         log of every change and experiment
```

The denoiser is built on the 3DGeoMeshNet graph auto-encoder; its original documentation is in [research/README_3DGeoMeshNet.md](research/README_3DGeoMeshNet.md).
