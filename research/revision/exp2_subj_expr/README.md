# Exp2: (subject, expression) split

Generalisation to unseen identities and unseen expressions. One model, trained on
S_train × E_train only, is tested on four cells:

| cell | subjects | expressions | test frames | meshes | samples |
|---|---|---|---|---|---|
| `ref` | seen | seen | official test frames (x91–x00) | 1,045 | 3,135 |
| `a` | **unseen** | seen | every 5th frame | 671 | 2,013 |
| `b` | seen | **unseen** | every 5th frame | 476 | 1,428 |
| `c` | **unseen** | **unseen** | every 2nd frame | 164 | 492 |

- Test subjects (random, seed 2026): `FaceTalk_170913_03279_TA`, `FaceTalk_170915_00223_TA`
- Test expressions (chosen as one moderate + one hard): `high_smile`, `mouth_extreme`
- Train + val: 10 subjects × 10 expressions = 13,430 meshes (80/20 inside `process()`, seed 1)
- Benchmarks use the same masks and format as `revision/benchmark` (3 mask types per mesh, 5–40% holes, seed 2026 + cell index)

Report each unseen cell **against `ref`** from this same model. This model sees about 30% less data than R1, so comparing with R1 would mix the data loss with the generalisation gap.

Caveat: the first frames of every COMA sequence are close to neutral, so the start of an "unseen expression" sequence is not really unseen. Frame-level CSVs make it possible to report only the peak part of each sequence if a reviewer asks.

## Isolation

Nothing outside this folder is written. `data/` and `revision/data` (the official-split cache used by R1) are untouched.

| Path | Contents |
|---|---|
| `raw_train/` | hard links to the S_train × E_train meshes in `data/train` (no copies) |
| `data/` | processed cache, `norm_completion.pt`, `template.ply` (mean of this training set only) |
| `splits/` | `split.json` (tracked in git) and file lists per set |
| `benchmark/{ref,a,b,c}/` | masks + index per cell |
| `checkpoints/`, `results/{ref,a,b,c}/`, `logs/` | training and evaluation outputs |

Config: `research/configs/config_diffusion_exp2_subjexpr.cfg`. Scripts: `research/revision/exp2_subj_expr/`.

## Leakage checks

`make_split.py` checks that (1) no training file has a test subject or a test expression, (2) no test file is the same file on disk as a training file, and (3) no test mesh has the same vertices as any training mesh. `check_processed.py` checks that all config paths stay inside this folder, that the processed counts match `raw_train/`, and that the normalisation mean and the template come from this training set.

## Run

```
(3dmesh) D:\COURSAI\3Dis> python research\revision\exp2_subj_expr\make_split.py
(3dmesh) D:\COURSAI\3Dis> python research\revision\process_official_split.py --config_file research\configs\config_diffusion_exp2_subjexpr.cfg
(3dmesh) D:\COURSAI\3Dis> python research\revision\exp2_subj_expr\check_processed.py
(3dmesh) D:\COURSAI\3Dis> research\revision\exp2_subj_expr\train_exp2.bat
(3dmesh) D:\COURSAI\3Dis> research\revision\exp2_subj_expr\test_exp2.bat
```

The first three steps are already done (2026-11-04). `make_split.py --check_only` reruns the leakage checks.
