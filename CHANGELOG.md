# Changelog

Newest first. Each entry lists code changes and experiments, with where outputs live (outputs themselves are not tracked).

## 2026-11-04: Official test benchmark + evaluation
- **Exp R1 training finished.** The user stopped it at epoch 289/300 because validation had plateaued; the best validation loss was 0.02847 at epoch 283.
- New `scripts/_revision/build_test_benchmark.py`: a fixed benchmark on the official COMA test set. 1,483 meshes × {random_scatter, bfs_patch, plane_cut} = 4,449 samples, hole sizes uniform in 5–40%, seed 2026. Output: `revision/benchmark/official_test_masks.npz` and `official_test_index.csv`. All methods must use these masks.
- New `scripts/_revision/eval_official_test.py`: missing/overall L2 (mm) and normal error in the hole and on the seam (deg). Writes a per-sample CSV, can resume after an interruption, and summarises by mask type and hole size.
- New `scripts/_revision/test_full_model.bat`: runs R1 on the benchmark. Output: `revision/results/official_test/predict_x0_depthw_nrm/`.
- **R1 results** (4,449 samples, best-validation EMA, single draw). Missing-region L2 in mm:

  | | all | ≤15% | 15–25% | >25% |
  |---|---|---|---|---|
  | scatter | 0.324 | 0.318 | 0.317 | 0.334 |
  | patch | 2.125 | 1.400 | 2.429 | 2.884 |
  | plane cut | 2.651 | 2.079 | 2.708 | 2.991 |
  | **all** | **1.700** | 1.263 | 1.748 | 2.057 |

  Normals: 5.36° in the hole, 2.76° on the seam. The old paper test (20 meshes, 5–15% holes, leaky split) gave 0.290 / 1.596 / 2.004. The like-for-like ≤15% bin gives 0.318 / 1.400 / 2.079, so the results hold on the leak-free split.
- Fix: hole-size bins in `eval_official_test.py` now cover all samples (13 masks fell just outside 5–40%).
- New `scripts/_revision/export_examples.py`: exports 20 benchmark samples (7 scatter, 7 patch, 6 plane cut, spanning hole sizes), each as GT / partial / completed / error-coloured mesh / mask. Output: `revision/results/official_test/predict_x0_depthw_nrm/examples/`.
- Observation: the worst plane cuts are small ones (5.1% → 5.0 mm, 12.3% → 5.6 mm). These holes include the mesh's open lower border (neck rim), so they are extrapolation with no observed geometry on the far side. Plane cuts away from the border (19–33%) give 1.1–1.9 mm.

## 2026-11-03: Repository set up
- Initial commit of the code base: the 3DGeoMeshNet base model, the diffusion completion model, baselines, evaluation, ablation and figure scripts.

## 2026-11-01: Official COMA train/test split
- The official COMA test subset (`data/COMA_test_subset/gt`, 1,483 meshes) was found to be 100% duplicated in `data/train` (identical vertices). All 1,483 were moved to `data/_revision/removed_from_train/`. The training folder now has 18,978 meshes.
- `config/config.py`: new optional `train_dir` key.
- `utils/completion_dataset.py`: `process()` reads from `train_dir` when set.
- New `config/config_diffusion_revision.cfg`: dataset root `revision/data`, checkpoints `revision/checkpoints/diffusion`.
- New `scripts/_revision/process_official_split.py`: built the cache with train 15,182 / val 3,796 (seed 1).
- New `scripts/_revision/train_full_model.bat`. **Exp R1:** the full model (x0 + depth weight + normal loss 0.01, 300 epochs) retrained on the official split. Output: `revision/checkpoints/diffusion/predict_x0_depthw_nrm/`. Status: training.
