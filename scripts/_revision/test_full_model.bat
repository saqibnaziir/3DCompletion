@echo off
REM ============================================================================
REM  Revision exp 1 (test): full model on the official COMA test benchmark.
REM  4,449 samples = 1,483 test meshes x {scatter, patch, plane cut}, 5-40% holes
REM  (masks fixed in revision/benchmark/, shared by every method).
REM
REM  Weights : best-validation EMA checkpoint
REM  Output  : revision/results/official_test/predict_x0_depthw_nrm/per_sample.csv
REM  Resumable: re-run the same command after an interruption.
REM
REM  Run from:  (3dmesh) D:\COURSAI\3Dis> scripts\_revision\test_full_model.bat
REM ============================================================================
set PYTHONIOENCODING=utf-8

python scripts\_revision\eval_official_test.py --param x0 --tag depthw_nrm --save_meshes 30 %*
