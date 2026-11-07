# 3D Mesh Completion via Conditional Offset Diffusion

Fills holes in registered 3D face meshes. A graph U-Net diffusion model generates the missing vertices, and the observed vertices are kept exactly as given. Trained and evaluated on [COMA](https://coma.is.tue.mpg.de/) (5,023-vertex faces with a fixed topology).

- **Offset diffusion:** the model generates each vertex's offset from the template (mean) face.
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

Download COMA (licence required) and put it in `data/COMA/`, keeping the original layout:

```
data/COMA/<subject>/<expression>/<expression>.<frame>.ply
```

See [`data/README.md`](data/README.md) for details. If the data is somewhere else, set `coma_dir` in the config.

## Quick start

```bash
python prepare_data.py   # 1. split, leakage checks, processed cache, test benchmark   (~15 min, CPU)
python train.py          # 2. train the model                                          (~2.5 days, 300 epochs)
python test.py           # 3. evaluate on the test set and print the results table
```

All three commands read [`config/coma.cfg`](config/coma.cfg). If training stops, continue it with `python train.py --resume`. For a quick check that everything works, run `python train.py --epochs 1` and then `python test.py --limit 10`.

## Configuration

Every setting is in **[`config/coma.cfg`](config/coma.cfg)**, with a comment on each one. The main ones:

| Setting | Default | Meaning |
|---|---|---|
| `coma_dir` | `data/COMA` | where the raw meshes are |
| `work_dir` | `runs/coma_subject_split` | where everything generated is written |
| `test_subjects` | 2 subjects | held out from training |
| `test_expressions` | *(empty)* | also hold out expressions; empty = subject split |
| `train_pair_fraction` | `1.0` | train on a random subset of (subject, expression) pairs |
| `run_name` | `full` | name of the checkpoint folder |
| `epoch`, `batch_size` | `300`, `4` | training length and batch size |

To run another experiment, copy the config, edit it (with its own `work_dir`), and pass it to every command, for example `python train.py --config config/my_experiment.cfg`.

## Evaluation protocol

- **Split:** the test subjects are never seen in training. The official COMA test frames ([`splits/coma_official_test.txt`](splits/coma_official_test.txt)) are also removed from training. The training data is shuffled 80/20 into train and validation; validation only selects the best checkpoint.
- **Test sets:**
  - `seen_subject`: training subjects, official test frames (the reference)
  - `unseen_subject`: held-out subjects, every 5th frame
  - `unseen_expression` and `unseen_both`: only if `test_expressions` is set
- **Holes:** every test mesh gets three fixed holes: scattered vertices, a contiguous patch and a plane cut. Each removes 5–40% of the vertices (seed 2026), and every method is scored on the same holes.
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

Training curves: `tensorboard --logdir runs/coma_subject_split/checkpoints`

## Project structure

```
├── prepare_data.py          step 1: data preparation
├── train.py                 step 2: training
├── test.py                  step 3: evaluation
├── config/
│   ├── coma.cfg             all settings
│   └── config.py            config parser
├── scripts/
│   ├── train_diffusion.py   training loop
│   ├── infer_diffusion.py   model loading and DDIM sampling with hard data consistency
│   └── evaluate_benchmark.py
├── utils/                   graph U-Net, diffusion process, dataset, masking, losses, metrics
├── splits/                  official COMA test list
├── data/                    put the data here
└── research/                scripts behind the paper's ablations and figures (not needed to run)
```

## Acknowledgements

The graph U-Net builds on the 3DGeoMeshNet mesh auto-encoder ([research/README_3DGeoMeshNet.md](research/README_3DGeoMeshNet.md)). COMA is from Ranjan et al., *Generating 3D faces using Convolutional Mesh Autoencoders*, ECCV 2018.
