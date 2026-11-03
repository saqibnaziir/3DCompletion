# Changelog

Newest first. Each entry lists code changes and experiments, with where outputs live (outputs themselves are not tracked).

## 2026-11-03: Repository set up
- Initial commit of the code base: the 3DGeoMeshNet base model, the diffusion completion model, baselines, evaluation, ablation and figure scripts.

## 2026-11-01: Official COMA train/test split
- The official COMA test subset (`data/COMA_test_subset/gt`, 1,483 meshes) was found to be 100% duplicated in `data/train` (identical vertices). All 1,483 were moved to `data/_revision/removed_from_train/`. The training folder now has 18,978 meshes.
- `config/config.py`: new optional `train_dir` key.
- `utils/completion_dataset.py`: `process()` reads from `train_dir` when set.
- New `config/config_diffusion_revision.cfg`: dataset root `revision/data`, checkpoints `revision/checkpoints/diffusion`.
- New `scripts/_revision/process_official_split.py`: built the cache with train 15,182 / val 3,796 (seed 1).
- New `scripts/_revision/train_full_model.bat`. **Exp R1:** the full model (x0 + depth weight + normal loss 0.01, 300 epochs) retrained on the official split. Output: `revision/checkpoints/diffusion/predict_x0_depthw_nrm/`. Status: training.
