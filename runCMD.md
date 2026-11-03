# Diffusion Mesh Completion — Command Reference

All commands run from `D:\COURSAI\3Dis` inside the `(3dmesh)` conda env.

The model is a graph-U-Net **diffusion** denoiser over the per-vertex deformation
(offset) field, conditioned on the partial mesh, with **hard known-region clamping**
at sampling (known vertices preserved exactly, no post-processing needed).

Core files:
- `config/config_diffusion.cfg`  — all hyperparameters
- `scripts/train_diffusion.py`   — training
- `scripts/infer_diffusion.py`   — evaluation / inference
- `scripts/complete_real_partial.py` — run on real cropped partials (test_old)
- `scripts/qualitative_figure.py`    — CVPR-style result figures

---

## 0. Environment

```
conda activate 3dmesh
cd D:\COURSAI\3Dis
```

> If you ever hit a `UnicodeEncodeError` on Windows, prefix the command with:
> `set PYTHONIOENCODING=utf-8` (run once per terminal session).

---

## 1. Training

**Quick sanity run (50 epochs) — confirm it learns:**
```
python scripts\train_diffusion.py --config_file config\config_diffusion.cfg --experiment_name diff_test --epochs 50
```

**Full run (300 epochs, recommended) — overnight, ~10 h at batch 8:**
```
python scripts\train_diffusion.py --config_file config\config_diffusion.cfg --experiment_name diff_v3_long --epochs 300 --batch_size 8
```

Notes:
- Best model is auto-saved on lowest eval loss, so `Ctrl+C` early is safe.
- Standardizes the target + saves `diffusion_scale.pt` (sigma + clip). Cosine LR decay is on.
- If the laptop gets sluggish / runs out of VRAM, lower the batch: `--batch_size 4` (or 2).
- Outputs go to `data\checkpoints_diffusion\` (overwrites previous run — back it up if you want to keep it).

Useful flags: `--batch_size N`, `--num_workers N`, `--epochs N`, `--seed N`,
`--experiment_name NAME`.

**Resuming after a crash / reboot:**

Every epoch writes a full `diffusion_state.pt` (model + EMA + optimizer + scheduler +
epoch + best_eval) and a line to `train_log.csv` in the arm's checkpoint dir. To pick
up where it stopped, re-run the **exact same command** with `--resume` appended:
```
python scripts\train_diffusion.py --config_file config\config_diffusion.cfg --epochs 300 --resume
```
A reboot then costs at most one epoch, and TensorBoard appends to the same run so the
curves stay continuous.

For runs started *before* `diffusion_state.pt` existed, only the weight files survive.
Bootstrap from them instead — this loses the Adam moment estimates (short re-warmup)
and rewinds to the last *best* epoch rather than the last completed one:
```
python scripts\train_diffusion.py ... --resume_from_weights --start_epoch 123 --best_eval 0.030155
```
`--start_epoch` is the saved weights' epoch + 1; it replays the cosine LR schedule to
that point. Always pass `--best_eval`, or the first epoch will overwrite the best
checkpoint with a worse one.

Watch training live:
```
tensorboard --logdir data\checkpoints_diffusion\tensorboard_logs
```

---

## 1b. Reconstruction (application — "one model, many tasks")

R1 "regenerate": condition the diffusion model on the **full** mesh (every vertex
known, mask=1 everywhere) and regenerate all 5023 vertices from noise **without**
hard clamping, then measure per-vertex L2 vs GT. Same trained model as completion
— no retraining, pure inference. Reconstructed `.ply` land in
`<test_dir>\reconstructed_diffusion\`.

```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --recon --test_dir data\completion_test_samples\test_set
```

Useful flags: `--steps N` (DDIM steps, default 50), `--clip F` (clamp guard),
`--num_samples N` (only the first is scored; extras save as diversity),
`--no_ema`. DFAUST: swap in `--data DFAUST` (drops `--config_file`/`--test_dir`).

---

## 2. Evaluation (quantitative + saves completed meshes)

**Evaluate the trained model on the test set** (samples currently live in `test_set\`):
```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test --test_dir data\completion_test_samples\test_set
```
Prints per-sample and aggregate missing-region / overall error, and writes completed
`.ply` to `<test_dir>\completed_diffusion\`.

If the GT/partial/mask files are at the default top level instead, just drop `--test_dir`:
```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test
```

**Diverse sampling + uncertainty (multiple completions per mesh):**
```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test --test_dir data\completion_test_samples\test_set --num_samples 5
```

**Sweep sampling knobs** (clamp is baked in at 6; override to experiment):
```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test --test_dir data\completion_test_samples\test_set --clip 4
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test --test_dir data\completion_test_samples\test_set --steps 200
```

Useful flags: `--steps N` (DDIM steps, default 50), `--clip F` (clamp, default 6),
`--num_samples N`, `--smooth` (optional boundary smoothing), `--no_ema` (raw weights).

---

## 3. Complete a single / real partial mesh

**One partial `.ply` (must be template topology, 5023 verts; missing = zeroed):**
```
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --input path\to\partial.ply --output path\to\completed.ply
```

**Real cropped partials in `data\test_old\partial\` (remapped via matching GT):**
```
python scripts\complete_real_partial.py --config_file config\config_diffusion.cfg
```

**Same, with diverse completions + uncertainty figures:**
```
python scripts\complete_real_partial.py --config_file config\config_diffusion.cfg --num_samples 5
```

Useful flags: `--clip F` (default 6), `--steps N`, `--num_samples N`,
`--partial_dir`, `--gt_dir`, `--out_dir`, `--view`, `--zoom`.

---

## 4. Visualization / result figures

**CVPR-style figure on the test set** — `GT | Input (partial) | Ours | Error`:
```
python scripts\qualitative_figure.py
```

**Pick specific samples / camera / error scale:**
```
python scripts\qualitative_figure.py --samples test_012 test_013 test_010 test_008 test_017 test_005
python scripts\qualitative_figure.py --view yz --zoom 1.6
python scripts\qualitative_figure.py --clim 0.012
```

**Calibrate one sample quickly (camera test):**
```
python scripts\qualitative_figure.py --single test_012
```

> `qualitative_figure.py` reads GT/partial from `data\completion_test_samples\test_set\`
> and completed meshes from `data\completion_test_samples\test_set\completed_diffusion\`
> (the default). Run the eval in Section 2 (with `--test_dir`) first so the completed
> meshes exist. Override paths with `--test_dir` / `--comp_dir` if your layout differs:
> ```
> python scripts\qualitative_figure.py --test_dir data\completion_test_samples\test_set
> ```

The **diversity + uncertainty** grid (the showcase figure) is produced by
`complete_real_partial.py --num_samples 5` (Section 3) at
`data\test_old\completed_diffusion\figures\diversity\_diversity_grid_largeholes.png`.

---

## 5. Where outputs land

| What | Path |
|------|------|
| Checkpoints (denoiser, EMA, scale) | `data\checkpoints_diffusion\` |
| TensorBoard logs | `data\checkpoints_diffusion\tensorboard_logs\` |
| Test-set completed meshes | `<test_dir>\completed_diffusion\` |
| Test-set figures | `data\completion_test_samples\qualitative_results\` |
| Real-partial completed meshes | `data\test_old\completed_diffusion\` |
| Real-partial figures + diversity | `data\test_old\completed_diffusion\figures\` |

---

## 6. Typical end-to-end sequence

```
:: 1) train
python scripts\train_diffusion.py --config_file config\config_diffusion.cfg --experiment_name diff_v3_long --epochs 300 --batch_size 8

:: 2) quantitative eval
python scripts\infer_diffusion.py --config_file config\config_diffusion.cfg --eval_test --test_dir data\completion_test_samples\test_set

:: 3) real partials + diversity/uncertainty
python scripts\complete_real_partial.py --config_file config\config_diffusion.cfg --num_samples 5

:: 4) test-set qualitative figure
python scripts\qualitative_figure.py
```

---

## 7. Debug / smoke tests (optional)

```
python scripts\_smoke_real.py    :: one batch end-to-end on real data + DDIM sample
python scripts\_smoke_scale.py   :: overfit + x0-recovery + clamped sampling sanity
```

---

## Current best result (300 epochs, clip 6)

- Test set (5–15% missing): **missing error 0.00213**, overall 0.00024
  (baseline V1/V2/V3 was 0.0049 → ~2.3x better, no post-processing).
- Real partials: robust up to ~49% missing (reconstructs half a face).

---

## 8. DFAUST full-body completion (`--data DFAUST`)

Same diffusion pipeline as COMA, on Dynamic FAUST bodies (SMPL topology, 6890
verts). Everything is switched with `--data COMA` vs `--data DFAUST` — that flag
just selects the config (`config\config_diffusion_dfaust.cfg`). Its own dataset
dir (`data\DFAUST_ds`), template, and checkpoints (`data\checkpoints_diffusion_dfaust`),
so it never clobbers the face model.

**One-time data prep** — export the HDF5 registrations to PLY (subsampled frames).
Needs `h5py` (only for this step). If it's not in the `3dmesh` env, run it with a
Python that has it (base anaconda), or `conda install -n 3dmesh h5py`:
```
python scripts\prepare_dfaust.py --stride 6
```
> `--stride 6` keeps every 6th frame of the 60 fps sequences → ~3.5k meshes.
> Lower stride = more meshes (more redundant), higher = fewer. Reads every
> `registrations_*.hdf5` in `data\DEFAUST\` (male + female if both present).

**Train:**
```
python scripts\train_diffusion.py --data DFAUST --experiment_name dfaust_long --epochs 400 --batch_size 4
```

**Evaluate on the held-out test bodies:**
```
python scripts\infer_diffusion.py --data DFAUST --eval_test
```

**Complete a single partial body** (template topology, 6890 verts, missing = zeroed):
```
python scripts\infer_diffusion.py --data DFAUST --input path\to\partial_body.ply --output completed_body.ply
```

**Diverse completions + per-vertex uncertainty (missing limbs):**
```
python scripts\infer_diffusion.py --data DFAUST --eval_test --num_samples 5
```

The COMA commands in Sections 1–4 are unchanged; add `--data COMA` (or keep the
explicit `--config_file config\config_diffusion.cfg`) to be explicit.

| DFAUST outputs | Path |
|------|------|
| Exported bodies (train PLYs) | `data\DFAUST_ds\train\<subject_seq>\*.ply` |
| Processed dataset cache + template | `data\DFAUST_ds\processed_completion\`, `data\DFAUST_ds\template.ply` |
| Held-out test samples (GT+partial+mask) | `data\DFAUST_ds\completion_test_samples\` |
| Checkpoints (denoiser, EMA, scale) | `data\checkpoints_diffusion_dfaust\` |

---

## 9. Paper contributions C2 / C4 (uncertainty benchmark + active rescanning)

Plan: `docs\paper\C2_C4_PLAN.md`. Both reuse the trained diffusion checkpoints —
no retraining. Add `--config_file config\config_diffusion.cfg` for COMA.

### C2 — correspondence-native uncertainty benchmark

Full run (DFAUST, all four mask regimes at once so the DoF analysis has range):
```
python scripts\eval_uncertainty_benchmark.py ^
  --config_file config\config_diffusion_dfaust.cfg ^
  --test_dirs data\DFAUST_ds\completion_test_samples ^
              data\DFAUST_ds\completion_test_samples_gentle ^
              data\DFAUST_ds\completion_test_samples_flat ^
              data\DFAUST_ds\completion_test_samples_plain ^
  --K 20 --regions 12 ^
  --out_dir data\DFAUST_ds\uncertainty_benchmark
```

COMA (note: COMA keeps its test meshes in a `test_set\` subfolder, DFAUST does
not — `--test_dir` is required here):
```
python scripts\eval_uncertainty_benchmark.py ^
  --config_file config\config_diffusion.cfg ^
  --test_dir data\completion_test_samples\test_set ^
  --K 20 --regions 10 ^
  --out_dir data\completion_test_samples\uncertainty_benchmark
```

Useful flags: `--no_setlevel` (skip the slow O(K^2) MMD/TMD/UHD block),
`--max_meshes N`, `--steps N` (fewer DDIM steps for a fast pass).

### C4 — uncertainty-guided next-best-scan

```
python scripts\eval_active_rescan.py ^
  --config_file config\config_diffusion_dfaust.cfg ^
  --test_dir data\DFAUST_ds\completion_test_samples ^
  --K 10 --rounds 5 --budget 0.10 --max_meshes 10 ^
  --out_dir data\DFAUST_ds\active_rescan
```

Ablation (scattered vertices instead of contiguous patches): add `--scatter`.

| C2 / C4 outputs | Path |
|------|------|
| C2 metrics, tables, plots | `<out_dir>\benchmark.json`, `baselines.csv`, `regions.csv`, `dof.csv`, `setlevel.csv`, `tables.tex`, `*.png` |
| C4 curves, summary, plot | `<out_dir>\rescan_curves.csv`, `rescan_summary.json`, `rescan.tex`, `rescan.png` |
