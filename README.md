# 3DCompletion: Conditional Offset Diffusion for Registered Mesh Completion

Completion of registered 3D meshes (fixed topology, dense correspondence to a template) with a conditional diffusion model. A multi-scale graph U-Net predicts per-vertex offsets from the template. At every DDIM step the observed vertices are re-imposed, so they come out exactly as given. Drawing several samples gives a per-vertex uncertainty map.

The denoiser is built on the **3DGeoMeshNet** graph auto-encoder (FeaStConv backbone, mesh down/up-sampling). The original base-model documentation is kept in [README_3DGeoMeshNet.md](README_3DGeoMeshNet.md).

## Main ideas

- **Offset diffusion:** the model diffuses the per-vertex displacement from the template mean shape. The target is standardized per axis (`diffusion_scale.pt`).
- **Hard data consistency:** observed vertices are re-imposed at every sampling step (replacement-based conditioning).
- **x0-prediction**, with an optional geodesic depth-weighted positional loss (deeper inside the hole means higher weight) and a surface-normal loss on the missing region.
- **Uncertainty:** the per-vertex spread over K samples, comparable vertex by vertex because correspondence is preserved.

## Repository layout

```
config/            experiment configs (*.cfg) + config parser
utils/             models, diffusion process, datasets, masking, losses, metrics
scripts/           training, inference, evaluation, baselines, ablations, figures
scripts/_revision/ scripts for the current revision experiments
Compattators/      wrappers for baselines (FaceCom, 3DGeoMeshNet completion)
Applications/      base-model applications (denoising, interpolation, extrapolation)
Mesh_completion/   base-model optimisation-based completion
err_cal/           error-computation helpers
revision/          README for the revision experiment folders (data/results are not tracked)
runCMD.md          full command reference
CHANGELOG.md       log of every change and experiment
```

Data, checkpoints, results, meshes and paper material are **not** tracked (see `.gitignore`).

## Setup

```bash
conda env create -f environment.yml
conda activate 3dmesh
```

Tested with Python 3.10, PyTorch 2.6 (CUDA 12.4) and PyTorch Geometric 2.6.1. PyTorch3D and pymeshlab are only needed for some baselines and figure scripts.

## Data

- **COMA** (faces, 5,023 vertices): https://coma.is.tue.mpg.de/
- **DFAUST** (bodies, 6,890 vertices): https://dfaust.is.tue.mpg.de/ (export with `scripts/prepare_dfaust.py`)

Both require signing their licences; no data is distributed here.

**Official COMA split (used from the revision onwards).** The test set is the standard COMA subset: frames x91–x00 of every 100-frame block, 1,483 meshes over all 12 subjects and 12 expressions. All of these meshes are removed from the training folder. The remaining 18,978 meshes are split 80/20 (seed 1) into train and validation. Validation is used only to select checkpoints.

```
data/train/<subject>/<expression>/*.ply     training meshes (test frames removed)
data/COMA_test_subset/gt/<subject>/*.ply    official test set
```

## Usage

All commands are run from the repository root. See [runCMD.md](runCMD.md) for the full list.

**Process data** (writes the cache, template and normalisation into the config's `dataset_dir`):
```bash
python scripts/_revision/process_official_split.py
```

**Train the full model** (x0 + geodesic depth weight + normal loss):
```bash
python scripts/train_diffusion.py --config_file config/config_diffusion_revision.cfg \
    --param x0 --depth_weight --depth_weight_normalize --lambda_normal 0.01 \
    --tag depthw_nrm --epochs 300 --batch_size 4 --seed 1
```
On Windows: `scripts\_revision\train_full_model.bat` (add `--resume` to continue a run).

**Evaluate / complete:**
```bash
python scripts/infer_diffusion.py --config_file config/config_diffusion.cfg --eval_test --test_dir <test_dir>
```

**Base model (3DGeoMeshNet auto-encoder):**
```bash
python scripts/train.py --config_file config/config.cfg
python scripts/reconstruction.py --config_file config/config.cfg --input_folder <in> --output_folder <out>
```

## Configs

| Config | Purpose |
|---|---|
| `config/config.cfg` | base 3DGeoMeshNet auto-encoder |
| `config/config_diffusion.cfg` | diffusion completion, original random 80/20 split |
| `config/config_diffusion_revision.cfg` | diffusion completion, **official COMA split** (writes to `revision/`) |
| `config/config_diffusion_dfaust.cfg` | diffusion completion on DFAUST |

`PATH_TO_PROJECT` in a config is replaced by the repository root. The optional `train_dir` key reads raw meshes from a different folder than `dataset_dir/train`.

## Experiment tracking

Every code change and experiment is committed, and summarised in [CHANGELOG.md](CHANGELOG.md).
