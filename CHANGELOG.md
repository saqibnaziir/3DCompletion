# Changelog

Newest first. Each entry lists code changes and experiments, with where outputs live (outputs themselves are not tracked).

## 2026-11-04: Official test benchmark + evaluation
- **Exp R1 training finished.** The user stopped it at epoch 289/300 because validation had plateaued; the best validation loss was 0.02847 at epoch 283.
- New `scripts/_revision/build_test_benchmark.py`: a fixed benchmark on the official COMA test set. 1,483 meshes × {random_scatter, bfs_patch, plane_cut} = 4,449 samples, hole sizes uniform in 5–40%, seed 2026. Output: `revision/benchmark/official_test_masks.npz` and `official_test_index.csv`. All methods must use these masks.
- New `scripts/_revision/eval_official_test.py`: missing/overall L2 (mm) and normal error in the hole and on the seam (deg). Writes a per-sample CSV, can resume after an interruption, and summarises by mask type and hole size.
- New `scripts/_revision/test_full_model.bat`: runs R1 on the benchmark. Output: `revision/results/official_test/predict_x0_depthw_nrm/`.

## 2026-11-03: Repository set up
- Initial commit of the code base: the 3DGeoMeshNet base model, the diffusion completion model, baselines, evaluation, ablation and figure scripts.

## 2026-11-01: Official COMA train/test split
- The official COMA test subset (`data/COMA_test_subset/gt`, 1,483 meshes) was found to be 100% duplicated in `data/train` (identical vertices). All 1,483 were moved to `data/_revision/removed_from_train/`. The training folder now has 18,978 meshes.
- `config/config.py`: new optional `train_dir` key.
- `utils/completion_dataset.py`: `process()` reads from `train_dir` when set.
- New `config/config_diffusion_revision.cfg`: dataset root `revision/data`, checkpoints `revision/checkpoints/diffusion`.
- New `scripts/_revision/process_official_split.py`: built the cache with train 15,182 / val 3,796 (seed 1).
- New `scripts/_revision/train_full_model.bat`. **Exp R1:** the full model (x0 + depth weight + normal loss 0.01, 300 epochs) retrained on the official split. Output: `revision/checkpoints/diffusion/predict_x0_depthw_nrm/`. Status: training.
