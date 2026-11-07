# 3D Mesh Completion via Conditional Offset Diffusion

Fills holes in registered 3D meshes. A graph U-Net diffusion model generates the missing vertices, and the observed vertices are kept exactly as given. Trained and evaluated on two datasets with a fixed mesh topology: [COMA](https://coma.is.tue.mpg.de/) faces (5,023 vertices) and [DFAUST](https://dfaust.is.tue.mpg.de/) bodies (6,890 vertices).

- **Offset diffusion:** the model generates each vertex's offset from the template (mean) shape.
- **Hard data consistency:** at every sampling step the observed vertices are put back, so the known part of the mesh is never changed.
- **Geometry-aware training:** the loss weight grows with geodesic depth inside the hole, and a surface-normal loss is added on the missing region.
- **Uncertainty:** drawing several samples gives a per-vertex uncertainty map.

## Requirements

| | |
|---|---|
| Python | 3.10 |
| GPU | NVIDIA, 8 GB is enough (batch size 4) |
| Main libraries | PyTorch 2.6, PyTorch Geometric 2.6.1, trimesh, quad-mesh-simplify (full list in [`requirements.txt`](requirements.txt)) |
| Tested on | Windows 11, CUDA 12.4, RTX 4060 Laptop GPU |

## Installation

```bash
git clone https://github.com/saqibnaziir/3DCompletion.git
cd 3DCompletion

conda create -n 3dmesh python=3.10 -y
conda activate 3dmesh
pip install -r requirements.txt
pip install --no-build-isolation quad-mesh-simplify==1.1.5
```

`quad-mesh-simplify` is compiled during installation, so it needs a C compiler: `build-essential` on Linux, or "Desktop development with C++" from the Visual Studio Build Tools on Windows. 

## Data

Both datasets need a licence from their authors, and neither is included here. Put them in `data/` (details in [`data/README.md`](data/README.md)):

| Dataset | Download | Put it in | Config |
|---|---|---|---|
| **COMA** (faces) | [coma.is.tue.mpg.de](https://coma.is.tue.mpg.de/) | `data/COMA/<subject>/<expression>/<expression>.<frame>.ply` (the original layout) | [`config/coma.cfg`](config/coma.cfg) |
| **DFAUST** (bodies) | [dfaust.is.tue.mpg.de](https://dfaust.is.tue.mpg.de/), the registrations | `data/DFAUST/registrations_m.hdf5` (and/or `registrations_f.hdf5`), as downloaded | [`config/dfaust.cfg`](config/dfaust.cfg) |

For DFAUST, `prepare_data.py` converts the HDF5 files to meshes itself, keeping every 6th frame. If your data is somewhere else, set `data_dir` in the config.

## Quick start

**COMA**
```bash
python prepare_data.py   # 1. split, leakage checks, processed cache, test benchmark   
python train.py          # 2. train the model                                          
python test.py           # 3. evaluate on the test set and print the results table
```

**DFAUST:** the same three commands with the DFAUST config:
```bash
python prepare_data.py --config config/dfaust.cfg
python train.py        --config config/dfaust.cfg
python test.py         --config config/dfaust.cfg
```

Without `--config`, the commands use [`config/coma.cfg`](config/coma.cfg). If training stops, continue it with `python train.py --resume` (plus the same `--config`). For a quick check that everything works, run `python train.py --epochs 1` and then `python test.py --limit 10`.

## Configuration

Every setting is in the config file ([`coma.cfg`](config/coma.cfg) or [`dfaust.cfg`](config/dfaust.cfg)), with a comment on each one. The main ones:

| Setting | COMA | DFAUST | Meaning |
|---|---|---|---|
| `data_dir` | `data/COMA` | `data/DFAUST` | where the raw data is |
| `work_dir` | `runs/coma_subject_split` | `runs/dfaust_subject_split` | where everything generated is written |
| `test_subjects` | 2 subjects | 1 subject (`50002`) | held out from training |
| `test_expressions` | *(empty)* | *(empty)* | also hold out expressions / motions; empty = subject split |
| `epoch` | 300 | 400 | training length |
| hole types | scatter, patch, plane cut | patch, plane cut | `strategies` in `[benchmark]` |
| `train_pair_fraction` | `1.0` | `1.0` | train on a random subset of (subject, sequence) pairs |
| `run_name` | `full` | `full` | name of the checkpoint folder |

To run another experiment, copy the config, edit it (with its own `work_dir`), and pass it to every command, for example `python train.py --config config/my_experiment.cfg`.

## Evaluation protocol

- **Split:** the test subjects are never seen in training. Some frames of the training subjects are also held out: for COMA the official test frames ([`splits/coma_official_test.txt`](splits/coma_official_test.txt)), and for DFAUST, which has no official split, the last 10% of every sequence. The rest is shuffled 80/20 into train and validation; validation only selects the best checkpoint.
- **Test sets:**
  - `seen_subject`: training subjects, held-out frames (the reference)
  - `unseen_subject`: held-out subjects (COMA every 5th frame, DFAUST every frame)
  - `unseen_expression` and `unseen_both`: only if `test_expressions` is set
- **Holes:** every test mesh gets fixed holes, one of each type, removing 5–40% of the vertices (seed 2026). The types are scattered vertices, a contiguous patch and a plane cut; DFAUST uses only the contiguous ones. Every method is scored on the same holes.
- **Metric:** mean per-vertex L2 error over the missing region, in mm. The angular normal error is also reported.
- **Leakage checks:** `prepare_data.py` stops if a test subject, expression or mesh (compared by vertex content) appears in the training data, or if the normalisation or template were not computed from the training set alone.

## Outputs

Everything is written to `work_dir`:

```
splits/split.json                         split definition and counts
data/                                     processed cache, template.ply, normalisation
benchmark/<test set>/                     the fixed test holes (masks.npz, index.csv)
checkpoints/predict_x0_<run_name>/        weights (best-validation EMA), train_log.csv, TensorBoard logs
results/<run_name>/<test set>/per_sample.csv   error for every test sample
results/<run_name>/summary.csv            results table
```

Training curves: `tensorboard --logdir runs/coma_subject_split/checkpoints` (or `runs/dfaust_subject_split/...`)

## Project structure

```
├── prepare_data.py          step 1: data preparation
├── train.py                 step 2: training
├── test.py                  step 3: evaluation
├── config/
│   ├── coma.cfg             all settings, COMA
│   ├── dfaust.cfg           all settings, DFAUST
│   └── config.py            config parser
├── scripts/
│   ├── train_diffusion.py   training loop
│   ├── infer_diffusion.py   model loading and DDIM sampling with hard data consistency
│   └── evaluate_benchmark.py
├── utils/                   graph U-Net, diffusion process, dataset, masking, losses, metrics
├── splits/                  official COMA test list
└── data/                    put the data here
```

## Acknowledgements

The graph U-Net builds on the 3DGeoMeshNet mesh auto-encoder. COMA is from Ranjan et al., *Generating 3D faces using Convolutional Mesh Autoencoders*, ECCV 2018. DFAUST is from Bogo et al., *Dynamic FAUST: Registering Human Bodies in Motion*, CVPR 2017.
