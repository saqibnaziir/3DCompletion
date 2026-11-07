# research/

Scripts behind the paper's experiments. **None of this is needed to prepare, train or test the model** (see the main README). The scripts here are kept as they were and run from the repository root, for example `python research/ablation_consistency.py ...`.

| What | Files |
|---|---|
| Earlier experiment configs | `configs/*.cfg` (`config_diffusion.cfg` = original COMA run, `config_diffusion_revision.cfg` = R1 on the official split, `config_diffusion_exp2_subjexpr.cfg` = Exp2) |
| Revision experiments | `revision/`: official-split benchmark, R1, depth deciles, K sweep, runtime, Exp2 (subject + expression split, in `revision/exp2_subj_expr/`) |
| Ablations | `ablation_diffusion.py`, `ablation_consistency.py`, `compare_arms.py`, `paired_arms.py`, `calibrate_lambda.py` |
| Baselines | `baseline_regressor.py` (deterministic graph U-Net, same backbone), `baselines_classical.py` (mean / Laplacian), `train_shapefusion.py`, `run_baseline_facecom.py`, `run_baseline_adapointr.py` |
| Evaluation and uncertainty | `eval_completion_dir.py`, `eval_coma_subset.py`, `eval_uncertainty_calibration.py`, `eval_uncertainty_benchmark.py`, `eval_active_rescan.py`, `diag_*.py` |
| Real partial scans | `complete_real_partial.py`, `analyze_real_partial.py` |
| Figures | `paper_figure_*.py`, `fig_uncertainty_hero.py`, `figure_half_missing.py`, `qualitative_figure.py`, `render_*.py`, `make_architecture_figure.py`, `dump_hero_samples.py` |
| DFAUST | `prepare_dfaust.py` (config: `configs/config_diffusion_dfaust.cfg`) |
| 3DGeoMeshNet base auto-encoder and old deterministic completion | `train_autoencoder.py`, `reconstruction.py`, `process.py`, `generate_samples.py`, `test_com.py`, `inpaint.py`, `train_completion.py`, `infer_completion.py`, `evaluate_completion.py` |

Files removed in the clean-up (smoke tests, one-off staging scripts, the old base-model applications, `runCMD.md`) are in git history under the tag `pre-cleanup`.
