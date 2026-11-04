# Revision status

Last updated: 2026-11-04. The detailed log is in [CHANGELOG.md](../CHANGELOG.md).

## Done
- [x] **Official COMA split.** The 1,483 test meshes were all duplicated in training, and have been removed. Train 15,182 / val 3,796 / test 1,483.
- [x] **Fixed test benchmark.** 4,449 samples = 1,483 meshes × {scatter, patch, plane cut}, 5–40% holes, seed 2026. Shared by all methods.
- [x] **R1: full model retrained and tested on the official split** (x0 + depth weight + normal loss, best-validation epoch 283).

## R1 results: missing-region L2 (mm), single draw

| | all | ≤15% | 15–25% | >25% |
|---|---|---|---|---|
| scatter | 0.324 | 0.318 | 0.317 | 0.334 |
| patch | 2.125 | 1.400 | 2.429 | 2.884 |
| plane cut | 2.651 | 2.079 | 2.708 | 2.991 |
| **all** | **1.700** | 1.263 | 1.748 | 2.057 |

Like-for-like with the old test (≤15% holes): scatter 0.290 → 0.318, patch 1.596 → 1.400, plane cut 2.004 → 2.079. **The results hold without the leak.** The worst plane cuts are the ones that touch the open neck border, where the model has to extrapolate.

## Next
1. Rerun the baselines on the same benchmark: 3DGeoMeshNet, FaceCom, Laplacian/biharmonic, and a deterministic U-Net (B2).
2. Subject-disjoint split for unseen identities (A1).
3. Multi-draw evaluation (mean ± std over draws), then the ablations, robustness and uncertainty calibration.
